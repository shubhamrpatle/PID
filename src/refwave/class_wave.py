"""Class-addressed point discovery with one shared visual/Qwen prefix.

LocateAnything normally serializes every requested class in one response. This
module keeps the same user-facing multi-class query but factorizes point discovery
internally: each requested class owns one branch, while all branches share exactly
one projected global image and one immutable Qwen prefix. Variable-length class
responses execute in bounded waves.
"""

from __future__ import annotations
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import torch
from .decoder import image_patch_lengths
from .lora import base_causal_lm
from .shared_prefix import SharedPrefixCache, legacy_key_values


@dataclass(frozen=True)
class ClassPointBranch:
    """One query-owned point-discovery branch."""

    label: str
    question: str
    seed: int


@dataclass
class _BranchState:
    branch: ClassPointBranch
    full_prompt_ids: list[int]
    suffix_ids: list[int]
    answer_ids: list[int]
    pending_ids: list[int]
    finished: bool = False
    fallback_reason: str | None = None
    forward_steps: int = 0


def longest_common_prefix(rows: Sequence[Sequence[int]]) -> list[int]:
    """Return the exact token prefix shared by every non-empty row."""
    if not rows or any((not row for row in rows)):
        raise ValueError("longest_common_prefix requires non-empty token rows")
    limit = min((len(row) for row in rows))
    cursor = 0
    while cursor < limit:
        value = int(rows[0][cursor])
        if any((int(row[cursor]) != value for row in rows[1:])):
            break
        cursor += 1
    return [int(value) for value in rows[0][:cursor]]


def _left_pad_rows(
    rows: Sequence[Sequence[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows or any((not row for row in rows)):
        raise ValueError("Every class branch must have a non-empty suffix")
    maximum = max((len(row) for row in rows))
    ids = torch.full((len(rows), maximum), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), maximum), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        values = torch.as_tensor(row, dtype=torch.long, device=device)
        ids[index, -len(row) :] = values
        mask[index, -len(row) :] = 1
    return (ids, mask)


def _right_pad_pending(
    states: Sequence[_BranchState], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    maximum = max((len(state.pending_ids) for state in states if not state.finished), default=0)
    if maximum == 0:
        return (
            torch.empty((len(states), 0), dtype=torch.long, device=device),
            torch.empty((len(states), 0), dtype=torch.long, device=device),
            0,
        )
    ids = torch.full((len(states), maximum), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros((len(states), maximum), dtype=torch.long, device=device)
    for index, state in enumerate(states):
        if state.finished or not state.pending_ids:
            continue
        values = torch.as_tensor(state.pending_ids, dtype=torch.long, device=device)
        ids[index, : len(state.pending_ids)] = values
        mask[index, : len(state.pending_ids)] = 1
    return (ids, mask, maximum)


class _BranchRNG:
    """Keep independent CPU/CUDA RNG streams invariant to wave width."""

    def __init__(self, seed: int, device: torch.device) -> None:
        self.device = device
        cpu_before = torch.random.get_rng_state()
        torch.manual_seed(int(seed))
        self.cpu_state = torch.random.get_rng_state()
        torch.random.set_rng_state(cpu_before)
        self.cuda_state: torch.Tensor | None = None
        if device.type == "cuda":
            cuda_before = torch.cuda.get_rng_state(device)
            torch.cuda.manual_seed(int(seed))
            self.cuda_state = torch.cuda.get_rng_state(device)
            torch.cuda.set_rng_state(cuda_before, device)

    def call(self, callback: Callable[[], Any]) -> Any:
        cpu_before = torch.random.get_rng_state()
        cuda_before = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        torch.random.set_rng_state(self.cpu_state)
        if self.cuda_state is not None:
            torch.cuda.set_rng_state(self.cuda_state, self.device)
        try:
            result = callback()
            self.cpu_state = torch.random.get_rng_state()
            if self.cuda_state is not None:
                self.cuda_state = torch.cuda.get_rng_state(self.device)
            return result
        finally:
            torch.random.set_rng_state(cpu_before)
            if cuda_before is not None:
                torch.cuda.set_rng_state(cuda_before, self.device)


class ClassAddressedPointWaveDecoder:
    """Generate one variable-length point set per semantic class in shared waves."""

    def __init__(
        self,
        worker: Any,
        *,
        image_token_limit: int = 6000,
        max_wave_size: int = 32,
        n_future_tokens: int = 6,
        bucket_suffix_lengths: bool = True,
    ) -> None:
        if max_wave_size < 1 or max_wave_size > 200:
            raise ValueError("max_wave_size must be in [1, 200]")
        self.worker = worker
        self.model = worker.model
        self.processor = worker.processor
        self.tokenizer = worker.tokenizer
        self.device = torch.device(worker.device)
        self.dtype = worker.dtype
        self.max_wave_size = int(max_wave_size)
        self.n_future_tokens = int(n_future_tokens)
        self.bucket_suffix_lengths = bool(bucket_suffix_lengths)
        if self.n_future_tokens != 6:
            raise ValueError("LocateAnything point discovery requires its native PBD6 window")
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None or not hasattr(image_processor, "in_token_limit"):
            raise RuntimeError("LocateAnything processor has no in_token_limit")
        image_processor.in_token_limit = int(image_token_limit)
        self.image_token_id = int(self.model.image_token_index)
        self.image_end_id = int(
            self.tokenizer.convert_tokens_to_ids(self.processor.image_end_token)
        )
        self.pad_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        self.mask_id = int(self.model.token_ids["default_mask_token_id"])
        model_module = sys.modules.get(self.model.__class__.__module__)
        if model_module is None:
            raise RuntimeError("LocateAnything model module is not loaded")
        self._sample_tokens = getattr(model_module, "sample_tokens")
        self._handle_pattern = getattr(model_module, "handle_pattern")

    def _message(self, image: Any, question: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
            }
        ]

    def _prepare_rows(
        self, image: Any, branches: Sequence[ClassPointBranch]
    ) -> tuple[list[list[int]], torch.Tensor]:
        first_message = self._message(image, branches[0].question)
        first_text = self.processor.py_apply_chat_template(
            first_message, tokenize=False, add_generation_prompt=True
        )
        (images, videos) = self.processor.process_vision_info(first_message)
        first = self.processor(text=[first_text], images=images, videos=videos, return_tensors="pt")
        lengths = image_patch_lengths(first["image_grid_hws"])
        if len(lengths) != 1:
            raise RuntimeError("Class-wave point discovery expects exactly one image")
        first_ids = [int(value) for value in first["input_ids"][0].tolist()]
        visual_count = first_ids.count(self.image_token_id)
        expected_visual_count = int(lengths[0]) // 4
        if visual_count != expected_visual_count:
            raise RuntimeError(
                f"Image placeholder mismatch: {visual_count} != {expected_visual_count}"
            )
        placeholder = (
            "<image 1>"
            + self.processor.image_start_token
            + self.processor.image_token * visual_count
            + self.processor.image_end_token
        )
        rows: list[list[int]] = []
        for branch in branches:
            text = self.processor.py_apply_chat_template(
                self._message(image, branch.question), tokenize=False, add_generation_prompt=True
            )
            if text.count("<image-1>") != 1:
                raise RuntimeError("Unexpected LocateAnything image placeholder layout")
            expanded = text.replace("<image-1>", placeholder)
            ids = self.tokenizer(expanded, add_special_tokens=True)["input_ids"]
            rows.append([int(value) for value in ids])
        if rows[0] != first_ids:
            raise RuntimeError(
                "Synthetic class-wave tokenization differs from the official processor"
            )
        pixels = first["pixel_values"].to(device=self.device, dtype=self.dtype)
        grids = torch.as_tensor(first["image_grid_hws"], dtype=torch.int32, device=self.device)
        features = self.model.extract_feature(pixels, grids)
        if len(features) != 1:
            raise RuntimeError("MoonViT returned an unexpected image count")
        projected_global = self.model.mlp1(features[0])
        return (rows, projected_global)

    def _split_rows(
        self, rows: Sequence[Sequence[int]], projected_global: torch.Tensor
    ) -> tuple[list[int], list[list[int]]]:
        first_row = [int(value) for value in rows[0]]
        image_positions = [
            index for (index, value) in enumerate(first_row) if int(value) == self.image_token_id
        ]
        if len(image_positions) != int(projected_global.shape[0]):
            raise RuntimeError("Class prompt does not contain the full visual grid")
        split = image_positions[-1] + 2
        if (
            not image_positions
            or split > len(first_row)
            or int(first_row[image_positions[-1] + 1]) != self.image_end_id
        ):
            raise RuntimeError("Class prompt does not contain a valid global image boundary")
        prefix = first_row[:split]
        suffixes: list[list[int]] = []
        for row in rows:
            normalized = [int(value) for value in row]
            if normalized[:split] != prefix:
                raise RuntimeError("Class branches do not share an identical image-bearing prefix")
            suffixes.append(normalized[split:])
        if any((not suffix for suffix in suffixes)):
            raise RuntimeError("Class-wave suffix is empty")
        return (prefix, suffixes)

    def _encode_prefix(
        self, prefix: Sequence[int], projected_global: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        ids = torch.as_tensor(prefix, dtype=torch.long, device=self.device).unsqueeze(0)
        causal = base_causal_lm(self.model.language_model)
        outputs = causal.model(
            input_ids=ids,
            visual_features=projected_global,
            image_token_index=self.image_token_id,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(len(prefix), device=self.device).unsqueeze(0),
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return legacy_key_values(outputs.past_key_values)

    def _sample_row(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        rng: _BranchRNG,
        generation_kwargs: Mapping[str, Any],
    ) -> tuple[str, list[int]]:

        def sample() -> tuple[Any, Any, Any, Any]:
            return self._sample_tokens(
                logits, generated, self.model.token_ids, keep_k=5, **dict(generation_kwargs)
            )

        (_probs, _confidence, x0, box_avg) = rng.call(sample)
        is_box_empty = bool((box_avg[0] == 0).all().item())
        candidate = x0[0] if is_box_empty else box_avg[0]
        pattern = self._handle_pattern(
            candidate, self.model.token_ids, str(generation_kwargs["generation_mode"])
        )
        return (str(pattern["type"]), [int(value) for value in pattern["tokens"]])

    def _independent_fallback(
        self,
        branch: ClassPointBranch,
        full_prompt_ids: Sequence[int],
        projected_global: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        generation_mode: str,
    ) -> str:
        """Finish a rare hybrid-AR branch without re-encoding the image.

        The released generator can switch a malformed PBD window to token-wise
        autoregression. Qwen's custom inference mask selects one mode for the
        whole batch, so mixed MTP/AR branches cannot remain in one wave. We rerun
        only those branches with the already projected global image. This keeps
        exact baseline behavior and records the extra Qwen prefill explicitly.
        """
        cpu_before = torch.random.get_rng_state()
        cuda_before = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        torch.manual_seed(int(branch.seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed(int(branch.seed))
        try:
            ids = torch.as_tensor(full_prompt_ids, dtype=torch.long, device=self.device).unsqueeze(
                0
            )
            response = self.model.generate(
                pixel_values=torch.empty(0, device=self.device, dtype=self.dtype),
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                visual_features=projected_global,
                image_grid_hws=None,
                tokenizer=self.tokenizer,
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
                generation_mode=generation_mode,
                decode_mode="baseline",
                temperature=float(temperature),
                do_sample=True,
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                verbose=False,
                return_debug_info=False,
            )
            return str(response[0] if isinstance(response, tuple) else response)
        finally:
            torch.random.set_rng_state(cpu_before)
            if cuda_before is not None:
                torch.cuda.set_rng_state(cuda_before, self.device)

    def _close_truncated_answer(self, answer: str, fallback_reason: str | None) -> tuple[str, bool]:
        """Close a max-length answer after its last complete geometry block.

        A branch that exhausts the token safety bound can end in the middle of a
        PBD block.  Keeping the malformed tail invents no useful geometry and
        breaks the public grammar.  We retain every completed block, discard only
        that incomplete tail, append the released EOS token, and expose the event
        in the result metadata.
        """
        eos = self.tokenizer.decode(
            [int(self.model.token_ids["im_end_token_id"])], skip_special_tokens=False
        )
        if str(answer).rstrip().endswith(eos):
            return (answer, False)
        if fallback_reason not in {"max_new_tokens", "hybrid_ar_required"}:
            return (answer, False)
        boundary = answer.rfind("</box>")
        if boundary < 0:
            return (answer, False)
        return (answer[: boundary + len("</box>")] + eos, True)

    def _decode_wave(
        self,
        prefix: Sequence[int],
        prefix_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        branches: Sequence[ClassPointBranch],
        full_rows: Sequence[Sequence[int]],
        suffix_rows: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        generation_mode: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if generation_mode not in {"fast", "hybrid"}:
            raise ValueError("Class-wave generation_mode must be fast or hybrid")
        states = [
            _BranchState(
                branch=branch,
                full_prompt_ids=[int(value) for value in full_row],
                suffix_ids=[int(value) for value in suffix],
                answer_ids=[],
                pending_ids=[],
            )
            for (branch, full_row, suffix) in zip(branches, full_rows, suffix_rows)
        ]
        rngs = [_BranchRNG(branch.seed, self.device) for branch in branches]
        (suffix_ids, suffix_mask) = _left_pad_rows(
            [state.suffix_ids for state in states], self.pad_id, self.device
        )
        batch = len(states)
        prefix_mask = torch.ones((batch, len(prefix)), dtype=torch.long, device=self.device)
        context_mask = torch.cat((prefix_mask, suffix_mask), dim=1)
        positions = context_mask.cumsum(-1) - 1
        positions.masked_fill_(context_mask == 0, 1)
        cache = SharedPrefixCache(prefix_key_values)
        causal = base_causal_lm(self.model.language_model)
        suffix_outputs = causal.model(
            input_ids=suffix_ids,
            visual_features=None,
            image_token_index=None,
            attention_mask=context_mask,
            position_ids=positions[:, -int(suffix_ids.shape[1]) :],
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        del suffix_outputs
        generation_kwargs = {
            "generation_mode": generation_mode,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
        }
        iterations = 0
        while not all((state.finished for state in states)):
            if max((len(state.answer_ids) for state in states)) >= int(max_new_tokens):
                for state in states:
                    if not state.finished:
                        state.finished = True
                        state.fallback_reason = "max_new_tokens"
                break
            iterations += 1
            (pending_ids, pending_mask, _pending_width) = _right_pad_pending(
                states, self.pad_id, self.device
            )
            last_tokens = []
            active_mask = []
            for state in states:
                if state.finished:
                    last_tokens.append(self.pad_id)
                    active_mask.append(0)
                else:
                    last_tokens.append(
                        state.pending_ids[-1] if state.pending_ids else state.suffix_ids[-1]
                    )
                    active_mask.append(1)
            mtp_ids = torch.full(
                (batch, self.n_future_tokens), self.mask_id, dtype=torch.long, device=self.device
            )
            mtp_ids[:, 0] = torch.as_tensor(last_tokens, dtype=torch.long, device=self.device)
            mtp_mask = (
                torch.as_tensor(active_mask, dtype=torch.long, device=self.device)
                .unsqueeze(1)
                .expand(-1, self.n_future_tokens)
            )
            model_ids = torch.cat((pending_ids, mtp_ids), dim=1)
            model_mask = torch.cat((pending_mask, mtp_mask), dim=1)
            full_mask = torch.cat((context_mask, model_mask), dim=1)
            model_positions = full_mask.cumsum(-1) - 1
            model_positions.masked_fill_(full_mask == 0, 1)
            model_positions[:, -self.n_future_tokens :] -= 1
            outputs = causal.model(
                input_ids=model_ids,
                visual_features=None,
                image_token_index=None,
                attention_mask=full_mask,
                position_ids=model_positions[:, -int(model_ids.shape[1]) :],
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state[:, -self.n_future_tokens :, :]
            logits = torch.nn.functional.linear(hidden, causal.lm_head.weight)
            cache.truncate_suffix(self.n_future_tokens)
            context_mask = torch.cat((context_mask, pending_mask), dim=1)
            histories = []
            maximum_history = max(
                (len(state.full_prompt_ids) + len(state.answer_ids) for state in states)
            )
            for state in states:
                row = state.full_prompt_ids + state.answer_ids
                histories.append([self.pad_id] * (maximum_history - len(row)) + row)
            generated = torch.as_tensor(histories, dtype=torch.long, device=self.device)
            for index, state in enumerate(states):
                if state.finished:
                    state.pending_ids = []
                    continue
                (out_type, tokens) = self._sample_row(
                    logits[index : index + 1],
                    generated[index : index + 1],
                    rngs[index],
                    generation_kwargs,
                )
                state.forward_steps += 1
                state.answer_ids.extend(tokens)
                state.pending_ids = tokens
                if out_type == "im_end":
                    state.finished = True
                elif out_type == "error_box":
                    state.finished = True
                    state.fallback_reason = "hybrid_ar_required"
        results = []
        for state in states:
            answer = self.tokenizer.decode(state.answer_ids, skip_special_tokens=False)
            results.append(
                {
                    "label": state.branch.label,
                    "question": state.branch.question,
                    "seed": int(state.branch.seed),
                    "answer": answer,
                    "tokens_generated": len(state.answer_ids),
                    "forward_steps": state.forward_steps,
                    "fallback_reason": state.fallback_reason,
                }
            )
        return (
            results,
            {
                "branches": len(states),
                "iterations": iterations,
                "fallbacks": sum((row["fallback_reason"] is not None for row in results)),
                "prefix_tokens": len(prefix),
                "maximum_suffix_tokens": max((len(row) for row in suffix_rows)),
            },
        )

    @torch.no_grad()
    def generate(
        self,
        image: Any,
        branches: Sequence[ClassPointBranch],
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        generation_mode: str = "hybrid",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not branches:
            return (
                [],
                {
                    "branches": 0,
                    "waves": 0,
                    "global_moonvit_encodes": 0,
                    "qwen_global_prefix_forwards": 0,
                    "end_to_end_seconds": 0.0,
                },
            )
        labels = [branch.label for branch in branches]
        if len(labels) != len(set(labels)):
            raise ValueError("Class-addressed branches must have unique labels")
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        (rows, projected_global) = self._prepare_rows(image, branches)
        (prefix, suffixes) = self._split_rows(rows, projected_global)
        prefix_key_values = self._encode_prefix(prefix, projected_global)
        indexed_output: dict[int, dict[str, Any]] = {}
        wave_stats: list[dict[str, Any]] = []
        if self.bucket_suffix_lengths:
            buckets: dict[int, list[int]] = {}
            for index, suffix in enumerate(suffixes):
                buckets.setdefault(len(suffix), []).append(index)
            wave_indices = [
                indices[start : start + self.max_wave_size]
                for (_length, indices) in sorted(buckets.items())
                for start in range(0, len(indices), self.max_wave_size)
            ]
        else:
            wave_indices = [
                list(range(start, min(len(branches), start + self.max_wave_size)))
                for start in range(0, len(branches), self.max_wave_size)
            ]
        for indices in wave_indices:
            (wave_output, stats) = self._decode_wave(
                prefix,
                prefix_key_values,
                [branches[index] for index in indices],
                [rows[index] for index in indices],
                [suffixes[index] for index in indices],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generation_mode=generation_mode,
            )
            stats["branch_indices"] = indices
            wave_suffix_lengths = [len(suffixes[index]) for index in indices]
            stats["suffix_tokens_min"] = min(wave_suffix_lengths)
            stats["suffix_tokens_max"] = max(wave_suffix_lengths)
            stats["suffix_token_lengths"] = wave_suffix_lengths
            wave_stats.append(stats)
            indexed_output.update(zip(indices, wave_output))
        output = [indexed_output[index] for index in range(len(branches))]
        fallback_count = 0
        for index, row in enumerate(output):
            if row["fallback_reason"] != "hybrid_ar_required":
                continue
            fallback_count += 1
            row["partial_wave_answer"] = row["answer"]
            row["answer"] = self._independent_fallback(
                branches[index],
                rows[index],
                projected_global,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generation_mode=generation_mode,
            )
            row["fallback_reused_projected_global"] = True
        syntax_closed_count = 0
        for row in output:
            (answer, closed) = self._close_truncated_answer(
                str(row["answer"]), row.get("fallback_reason")
            )
            row["answer"] = answer
            row["syntax_closed_at_token_bound"] = closed
            syntax_closed_count += int(closed)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak = float(torch.cuda.max_memory_allocated(self.device) / 1024**2)
        else:
            peak = 0.0
        elapsed = time.perf_counter() - started
        return (
            output,
            {
                "architecture": "class_addressed_shared_prefix_point_wave",
                "branches": len(branches),
                "requested_wave_size": self.max_wave_size,
                "effective_wave_sizes": [row["branches"] for row in wave_stats],
                "waves": len(wave_stats),
                "global_moonvit_encodes": 1,
                "qwen_global_prefix_forwards": 1,
                "qwen_branch_prefill_forwards": len(wave_stats),
                "qwen_generation_iterations": sum((row["iterations"] for row in wave_stats)),
                "fallbacks": fallback_count,
                "fallback_qwen_full_prompt_forwards": fallback_count,
                "suffix_length_bucketing": self.bucket_suffix_lengths,
                "suffix_length_buckets": sorted({len(row) for row in suffixes}),
                "syntax_closed_at_token_bound": syntax_closed_count,
                "shared_prefix_tokens": len(prefix),
                "projected_global_tokens": int(projected_global.shape[0]),
                "peak_gpu_mb": peak,
                "end_to_end_seconds": float(elapsed),
                "wave_stats": wave_stats,
            },
        )
