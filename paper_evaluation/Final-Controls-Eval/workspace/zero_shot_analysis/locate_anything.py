"""LocateAnything-3B integration helpers.

The imports are intentionally lazy so the rest of the evaluator remains usable
without the heavy VLM dependency stack.
"""

from __future__ import annotations

import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .schema import EvalSample, normalize_class_name


BOX_TOKEN_RE = re.compile(r"<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>", re.IGNORECASE)
POINT_TOKEN_RE = re.compile(r"<box>\s*<(\d+)>\s*<(\d+)>\s*</box>", re.IGNORECASE)
REF_OR_BOX_TOKEN_RE = re.compile(
    r"<ref>\s*(.*?)\s*</ref>|<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>",
    re.IGNORECASE | re.DOTALL,
)
REF_OR_POINT_OR_BOX_TOKEN_RE = re.compile(
    r"<ref>\s*(.*?)\s*</ref>|<box>\s*<(\d+)>\s*<(\d+)>(?:\s*<(\d+)>\s*<(\d+)>)?\s*</box>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class LocateAnythingResult:
    answer: str
    pred_hboxes: List[Dict[str, Any]]
    latency_ms: float
    tokens_generated: Optional[int]
    gpu_memory_mb: Optional[float]
    raw_stats: Dict[str, Any]


class GeometrySafeRepetitionPenaltyLogitsProcessor:
    """Apply repetition penalty only to non-geometry tokens.

    LocateAnything boxes are text tokens. Penalizing repeated coordinate and
    markup tokens can suppress dense scenes where many boxes need similar
    syntax. This processor mirrors HuggingFace's repetition penalty behavior
    but skips known box/ref/coordinate/special-token ids.
    """

    def __init__(self, penalty: float, safe_token_ids: Iterable[int]) -> None:
        if penalty <= 0:
            raise ValueError("penalty must be > 0")
        self.penalty = float(penalty)
        self.safe_token_ids: Set[int] = {int(token_id) for token_id in safe_token_ids}

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.penalty == 1.0:
            return scores
        for batch_idx in range(input_ids.shape[0]):
            seen = set(int(token_id) for token_id in input_ids[batch_idx].tolist())
            penalized = [token_id for token_id in seen if token_id not in self.safe_token_ids]
            if not penalized:
                continue
            token_ids = scores.new_tensor(penalized, dtype=input_ids.dtype)
            token_scores = scores[batch_idx, token_ids]
            scores[batch_idx, token_ids] = self.torch_where(token_scores < 0, token_scores * self.penalty, token_scores / self.penalty)
        return scores

    @staticmethod
    def torch_where(condition: Any, x: Any, y: Any) -> Any:
        import torch

        return torch.where(condition, x, y)


class EOSDelayLogitsProcessor:
    """Block EOS briefly unless the generated text has emitted enough boxes."""

    def __init__(
        self,
        tokenizer: Any,
        eos_token_ids: Iterable[int],
        min_new_tokens: int = 64,
        min_boxes_before_eos: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.eos_token_ids = [int(token_id) for token_id in eos_token_ids if token_id is not None]
        self.min_new_tokens = int(min_new_tokens)
        self.min_boxes_before_eos = int(min_boxes_before_eos)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if not self.eos_token_ids:
            return scores
        for batch_idx in range(input_ids.shape[0]):
            generated_ids = input_ids[batch_idx].tolist()
            if len(generated_ids) >= self.min_new_tokens:
                continue
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            if text.count("<box>") >= self.min_boxes_before_eos:
                continue
            for eos_id in self.eos_token_ids:
                if 0 <= eos_id < scores.shape[-1]:
                    scores[batch_idx, eos_id] = -math.inf
        return scores


def geometry_safe_token_ids(tokenizer: Any) -> set[int]:
    """Build the token-id allowlist for geometry-safe repetition penalties."""

    safe_strings = ["<box>", "</box>", "<ref>", "</ref>", "</c>"]
    safe_strings.extend(f"<{idx}>" for idx in range(1001))
    safe_ids: set[int] = set()

    for text in safe_strings:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        safe_ids.update(int(token_id) for token_id in token_ids)

    for token_id in [
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "unk_token_id", None),
        getattr(tokenizer, "mask_token_id", None),
    ]:
        if token_id is not None:
            safe_ids.add(int(token_id))

    for _token, token_id in getattr(tokenizer, "added_tokens_encoder", {}).items():
        safe_ids.add(int(token_id))
    return safe_ids


def eos_token_ids(tokenizer: Any) -> list[int]:
    values = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, Sequence) and not isinstance(eos_id, (str, bytes)):
        values.extend(int(token_id) for token_id in eos_id if token_id is not None)
    elif eos_id is not None:
        values.append(int(eos_id))
    for token in ["<|endoftext|>", "<|im_end|>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            values.append(int(token_id))
    return sorted(set(values))


def locateanything_prompt(sample: EvalSample) -> str:
    target = normalize_class_name(sample.target_class)
    if sample.query_type == "relation":
        return f"Locate all the instances that match the following description: {sample.query}"
    if sample.query_type == "negative":
        return f"Locate all the instances that matches the following description: {target}."
    if sample.query_type == "dense":
        return f"Locate all the instances that matches the following description: {target}."
    return f"Locate all the instances that matches the following description: {target}."


def parse_locateanything_boxes(answer: str, image_width: int, image_height: int, label: Optional[str] = None) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    for match in BOX_TOKEN_RE.finditer(answer):
        x1, y1, x2, y2 = [int(value) for value in match.groups()]
        px1 = max(0.0, min(float(image_width), x1 / 1000.0 * image_width))
        py1 = max(0.0, min(float(image_height), y1 / 1000.0 * image_height))
        px2 = max(0.0, min(float(image_width), x2 / 1000.0 * image_width))
        py2 = max(0.0, min(float(image_height), y2 / 1000.0 * image_height))
        boxes.append(
            {
                "hbox": [px1, py1, px2, py2],
                "score": 1.0,
                "label": label,
            }
        )
    return boxes


def parse_locateanything_labeled_boxes(
    answer: str,
    image_width: int,
    image_height: int,
    default_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Parse LocateAnything output while preserving the active <ref> label.

    Detection prompts can include many categories joined by ``</c>``. In that
    mode the model emits blocks such as ``<ref>ship</ref><box>...</box>``.
    Keeping the ref label is required for detector-style mAP.
    """

    boxes: List[Dict[str, Any]] = []
    current_label = default_label
    for match in REF_OR_BOX_TOKEN_RE.finditer(answer):
        ref_label = match.group(1)
        if ref_label is not None:
            current_label = normalize_class_name(ref_label)
            continue

        x1, y1, x2, y2 = [int(value) for value in match.groups()[1:]]
        px1 = max(0.0, min(float(image_width), x1 / 1000.0 * image_width))
        py1 = max(0.0, min(float(image_height), y1 / 1000.0 * image_height))
        px2 = max(0.0, min(float(image_width), x2 / 1000.0 * image_width))
        py2 = max(0.0, min(float(image_height), y2 / 1000.0 * image_height))
        boxes.append(
            {
                "hbox": [px1, py1, px2, py2],
                "score": 1.0,
                "label": current_label,
            }
        )
    return boxes


def parse_locateanything_labeled_points(
    answer: str,
    image_width: int,
    image_height: int,
    default_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Parse native LA point outputs while preserving the active <ref> label.

    LA represents pointing as ``<box><x><y></box>``. If a four-coordinate box is
    present, this parser uses its center as a fallback point, which mirrors the
    balanced-100 pointing evaluator.
    """

    points: List[Dict[str, Any]] = []
    current_label = default_label
    for match in REF_OR_POINT_OR_BOX_TOKEN_RE.finditer(answer):
        ref_label = match.group(1)
        if ref_label is not None:
            current_label = normalize_class_name(ref_label)
            continue
        x1 = int(match.group(2))
        y1 = int(match.group(3))
        x2_raw = match.group(4)
        y2_raw = match.group(5)
        if x2_raw is None or y2_raw is None:
            x_norm = float(x1)
            y_norm = float(y1)
            source = "point"
        else:
            x_norm = (float(x1) + float(x2_raw)) / 2.0
            y_norm = (float(y1) + float(y2_raw)) / 2.0
            source = "box_center_fallback"
        px = max(0.0, min(float(image_width), x_norm / 1000.0 * image_width))
        py = max(0.0, min(float(image_height), y_norm / 1000.0 * image_height))
        points.append(
            {
                "point": [px, py],
                "score": 1.0,
                "label": current_label,
                "source": source,
            }
        )
    return points


class LocateAnythingWorker:
    def __init__(self, model_path: str = "nvidia/LocateAnything-3B", device: str = "cuda", dtype: str = "bfloat16") -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LocateAnything dependencies are missing. Install requirements-locateanything.txt "
                "and a CUDA-compatible PyTorch build first."
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.torch = torch
        self.device = device
        self.dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config._attn_implementation = "sdpa"
        config._attn_implementation_autoset = False
        config.text_config._attn_implementation = "sdpa"
        config.text_config._attn_implementation_autoset = False
        # Qwen's custom LocateAnything code implements the released block mask
        # for SDPA. MoonViT can retain its efficient frozen inference backend.
        config.vision_config._attn_implementation = "flash_attention_2"
        config.vision_config._attn_implementation_autoset = False
        self.model = AutoModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device).eval()
        environment = __import__("os").environ
        pilot_adapter = environment.get("LOCANY_POINTRELAY_ADAPTER", "").strip()
        evidence_adapter = environment.get("LOCANY_EVIDENCE_ADAPTER", "").strip()
        witness_adapter = environment.get("LOCANY_WITNESS_ADAPTER", "").strip()
        adaptive_memory_adapter = environment.get("LOCANY_ADAPTIVE_MEMORY_ADAPTER", "").strip()
        dense_address_adapter = environment.get("LOCANY_DENSE_ADDRESS_ADAPTER", "").strip()
        marked_residual_adapter = environment.get("LOCANY_MARKED_RESIDUAL_ADAPTER", "").strip()
        latent_address_adapter = environment.get("LOCANY_LATENT_ADDRESS_ADAPTER", "").strip()
        distribution_address_adapter = environment.get(
            "LOCANY_DISTRIBUTION_ADDRESS_ADAPTER", ""
        ).strip()
        residual_set_adapter = environment.get(
            "LOCANY_RESIDUAL_SET_REENTRY_ADAPTER", ""
        ).strip()
        residual_geometry_adapter = environment.get(
            "LOCANY_RESIDUAL_GEOMETRY_REENTRY_ADAPTER", ""
        ).strip()
        evidence_sentinel_adapter = environment.get(
            "LOCANY_EVIDENCE_SENTINEL_REENTRY_ADAPTER", ""
        ).strip()
        latent_point_process_adapter = environment.get(
            "LOCANY_LATENT_POINT_PROCESS_REENTRY_ADAPTER", ""
        ).strip()
        posterior_guided_visual_event_adapter = environment.get(
            "LOCANY_POSTERIOR_GUIDED_VISUAL_EVENT_ADAPTER", ""
        ).strip()
        posterior_distilled_visual_event_adapter = environment.get(
            "LOCANY_POSTERIOR_DISTILLED_VISUAL_EVENT_ADAPTER", ""
        ).strip()
        exchangeable_visual_event_adapter = environment.get(
            "LOCANY_EXCHANGEABLE_VISUAL_EVENT_ADAPTER", ""
        ).strip()
        geometry_coupled_visual_event_adapter = environment.get(
            "LOCANY_GEOMETRY_COUPLED_VISUAL_EVENT_ADAPTER", ""
        ).strip()
        accepted_mark_distilled_visual_event_adapter = environment.get(
            "LOCANY_ACCEPTED_MARK_DISTILLED_VISUAL_EVENT_ADAPTER", ""
        ).strip()
        mass_conserving_visual_event_adapter = environment.get(
            "LOCANY_MASS_CONSERVING_VISUAL_EVENT_ADAPTER", ""
        ).strip()
        open_ended_residual_event_adapter = environment.get(
            "LOCANY_OPEN_ENDED_RESIDUAL_EVENT_ADAPTER", ""
        ).strip()
        point_marked_residual_event_adapter = environment.get(
            "LOCANY_POINT_MARKED_RESIDUAL_EVENT_ADAPTER", ""
        ).strip()
        intensity_guided_visual_event_adapter = environment.get(
            "LOCANY_INTENSITY_GUIDED_VISUAL_EVENT_ADAPTER", ""
        ).strip()
        semantic_aligned_event_adapter = environment.get(
            "LOCANY_SEMANTIC_ALIGNED_EVENT_ADAPTER", ""
        ).strip()
        pivr_la_adapter = environment.get(
            "LOCANY_PIVR_LA_SINGLE_PASS_ADAPTER", ""
        ).strip()
        legacy_ravel_adapter = environment.get("LOCANY_RAVEL_ADAPTER", "").strip()
        if pivr_la_adapter and legacy_ravel_adapter:
            raise RuntimeError(
                "Set only LOCANY_PIVR_LA_SINGLE_PASS_ADAPTER; "
                "LOCANY_RAVEL_ADAPTER is its legacy alias"
            )
        ravel_adapter = pivr_la_adapter or legacy_ravel_adapter
        if sum(
            bool(value)
            for value in (
                pilot_adapter,
                evidence_adapter,
                witness_adapter,
                adaptive_memory_adapter,
                dense_address_adapter,
                marked_residual_adapter,
                latent_address_adapter,
                distribution_address_adapter,
                residual_set_adapter,
                residual_geometry_adapter,
                evidence_sentinel_adapter,
                latent_point_process_adapter,
                posterior_guided_visual_event_adapter,
                posterior_distilled_visual_event_adapter,
                exchangeable_visual_event_adapter,
                geometry_coupled_visual_event_adapter,
                accepted_mark_distilled_visual_event_adapter,
                mass_conserving_visual_event_adapter,
                open_ended_residual_event_adapter,
                point_marked_residual_event_adapter,
                intensity_guided_visual_event_adapter,
                semantic_aligned_event_adapter,
                ravel_adapter,
            )
        ) > 1:
            raise RuntimeError("Only one localization pilot adapter can be active")
        self.pointrelay_pilot = None
        self.semantic_aligned_event_pilot = None
        self.evidence_pilot = None
        self.witness_pilot = None
        self.adaptive_memory_pilot = None
        self.dense_address_pilot = None
        self.marked_residual_pilot = None
        self.latent_address_pilot = None
        self.distribution_address_pilot = None
        self.residual_set_pilot = None
        self.residual_geometry_pilot = None
        self.evidence_sentinel_pilot = None
        self.latent_point_process_pilot = None
        self.posterior_guided_visual_event_pilot = None
        self.posterior_distilled_visual_event_pilot = None
        self.exchangeable_visual_event_pilot = None
        self.geometry_coupled_visual_event_pilot = None
        self.accepted_mark_distilled_visual_event_pilot = None
        self.mass_conserving_visual_event_pilot = None
        self.open_ended_residual_event_pilot = None
        self.point_marked_residual_event_pilot = None
        self.intensity_guided_visual_event_pilot = None
        self.ravel_pilot = None
        if ravel_adapter:
            from .ravel_pbd import load_pivr_la_single_pass_adapter

            self.ravel_pilot = load_pivr_la_single_pass_adapter(
                self.model, ravel_adapter
            )
            print(
                "Loaded PIVR-LA single-pass adapter: "
                f"{ravel_adapter} "
                f"(arm={self.ravel_pilot['checkpoint']['config']['arm']})"
            )
        elif semantic_aligned_event_adapter:
            from .semantic_aligned_event_reentry_pbd import (
                load_semantic_aligned_event_reentry_adapter,
            )

            self.semantic_aligned_event_pilot = (
                load_semantic_aligned_event_reentry_adapter(
                    self.model, semantic_aligned_event_adapter
                )
            )
            print(
                "Loaded semantic-aligned recurrent-event adapter: "
                f"{semantic_aligned_event_adapter} "
                f"(arm={self.semantic_aligned_event_pilot['checkpoint']['config']['arm']})"
            )
        elif pilot_adapter:
            from .pointrelay_pbd import load_pilot_adapter

            self.pointrelay_pilot = load_pilot_adapter(self.model, pilot_adapter)
            print(
                "Loaded PointRelay pilot adapter: "
                f"{pilot_adapter} (arm={self.pointrelay_pilot['checkpoint']['config']['arm']})"
            )
        elif evidence_adapter:
            from .residual_evidence_pbd import load_evidence_adapter

            self.evidence_pilot = load_evidence_adapter(self.model, evidence_adapter)
            print(
                "Loaded residual-evidence pilot adapter: "
                f"{evidence_adapter} (arm={self.evidence_pilot['checkpoint']['config']['arm']})"
            )
        elif witness_adapter:
            from .witness_pbd import load_witness_adapter

            self.witness_pilot = load_witness_adapter(self.model, witness_adapter)
            print(
                "Loaded WitnessRoute-PBD adapter: "
                f"{witness_adapter} "
                f"(arm={self.witness_pilot['checkpoint']['config']['arm']})"
            )
        elif adaptive_memory_adapter:
            from .adaptive_memory_pbd import load_adaptive_memory_adapter

            self.adaptive_memory_pilot = load_adaptive_memory_adapter(
                self.model, adaptive_memory_adapter
            )
            print(
                "Loaded adaptive point-memory adapter: "
                f"{adaptive_memory_adapter} "
                f"(arm={self.adaptive_memory_pilot['checkpoint']['config']['arm']})"
            )
        elif dense_address_adapter:
            from .dense_address_field_pbd import load_dense_address_field_adapter

            self.dense_address_pilot = load_dense_address_field_adapter(
                self.model, dense_address_adapter
            )
            print(
                "Loaded dense visual-address adapter: "
                f"{dense_address_adapter} "
                f"(arm={self.dense_address_pilot['checkpoint']['config']['arm']})"
            )
        elif marked_residual_adapter:
            from .marked_residual_address_pbd import load_marked_residual_address_adapter

            self.marked_residual_pilot = load_marked_residual_address_adapter(
                self.model, marked_residual_adapter
            )
            print(
                "Loaded marked-residual visual re-entry adapter: "
                f"{marked_residual_adapter} "
                f"(arm={self.marked_residual_pilot['checkpoint']['config']['arm']})"
            )
        elif latent_address_adapter:
            from .latent_address_reentry_pbd import load_latent_address_adapter

            self.latent_address_pilot = load_latent_address_adapter(
                self.model, latent_address_adapter
            )
            print(
                "Loaded latent-address visual re-entry adapter: "
                f"{latent_address_adapter} "
                f"(arm={self.latent_address_pilot['checkpoint']['config']['arm']})"
            )
        elif distribution_address_adapter:
            from .distribution_addressed_reentry_pbd import (
                load_distribution_addressed_adapter,
            )

            self.distribution_address_pilot = load_distribution_addressed_adapter(
                self.model, distribution_address_adapter
            )
            print(
                "Loaded distribution-addressed visual re-entry adapter: "
                f"{distribution_address_adapter} "
                f"(arm={self.distribution_address_pilot['checkpoint']['config']['arm']})"
            )
        elif residual_set_adapter:
            from .residual_set_reentry_pbd import load_residual_set_reentry_adapter

            self.residual_set_pilot = load_residual_set_reentry_adapter(
                self.model, residual_set_adapter
            )
            print(
                "Loaded residual-set visual re-entry adapter: "
                f"{residual_set_adapter} "
                f"(arm={self.residual_set_pilot['checkpoint']['config']['arm']})"
            )
        elif residual_geometry_adapter:
            from .residual_geometry_reentry_pbd import (
                load_residual_geometry_reentry_adapter,
            )

            self.residual_geometry_pilot = load_residual_geometry_reentry_adapter(
                self.model, residual_geometry_adapter
            )
            print(
                "Loaded residual-geometry visual re-entry adapter: "
                f"{residual_geometry_adapter} "
                f"(arm={self.residual_geometry_pilot['checkpoint']['config']['arm']})"
            )
        elif evidence_sentinel_adapter:
            from .evidence_sentinel_reentry_pbd import (
                load_evidence_sentinel_reentry_adapter,
            )

            self.evidence_sentinel_pilot = load_evidence_sentinel_reentry_adapter(
                self.model, evidence_sentinel_adapter
            )
            print(
                "Loaded evidence-sentinel visual re-entry adapter: "
                f"{evidence_sentinel_adapter} "
                f"(arm={self.evidence_sentinel_pilot['checkpoint']['config']['arm']})"
            )
        elif latent_point_process_adapter:
            from .latent_point_process_reentry_pbd import (
                load_latent_point_process_reentry_adapter,
            )

            self.latent_point_process_pilot = load_latent_point_process_reentry_adapter(
                self.model, latent_point_process_adapter
            )
            print(
                "Loaded latent-point-process visual re-entry adapter: "
                f"{latent_point_process_adapter} "
                f"(arm={self.latent_point_process_pilot['checkpoint']['config']['arm']})"
            )
        elif point_marked_residual_event_adapter:
            from .point_marked_residual_event_pbd import (
                load_point_marked_residual_event_adapter,
            )

            self.point_marked_residual_event_pilot = (
                load_point_marked_residual_event_adapter(
                    self.model, point_marked_residual_event_adapter
                )
            )
            print(
                "Loaded point-marked residual-event adapter: "
                f"{point_marked_residual_event_adapter} "
                f"(arm={self.point_marked_residual_event_pilot['checkpoint']['config']['arm']})"
            )
        elif open_ended_residual_event_adapter:
            from .open_ended_residual_event_pbd import (
                load_open_ended_residual_event_adapter,
            )

            self.open_ended_residual_event_pilot = (
                load_open_ended_residual_event_adapter(
                    self.model, open_ended_residual_event_adapter
                )
            )
            print(
                "Loaded open-ended residual-event adapter: "
                f"{open_ended_residual_event_adapter} "
                f"(arm={self.open_ended_residual_event_pilot['checkpoint']['config']['arm']})"
            )
        elif mass_conserving_visual_event_adapter:
            from .mass_conserving_event_reentry_pbd import (
                load_mass_conserving_visual_event_adapter,
            )

            self.mass_conserving_visual_event_pilot = (
                load_mass_conserving_visual_event_adapter(
                    self.model, mass_conserving_visual_event_adapter
                )
            )
            print(
                "Loaded mass-conserving visual-event adapter: "
                f"{mass_conserving_visual_event_adapter} "
                f"(arm={self.mass_conserving_visual_event_pilot['checkpoint']['config']['arm']})"
            )
        elif accepted_mark_distilled_visual_event_adapter:
            from .accepted_mark_distilled_event_reentry_pbd import (
                load_accepted_mark_distilled_visual_event_adapter,
            )

            self.accepted_mark_distilled_visual_event_pilot = (
                load_accepted_mark_distilled_visual_event_adapter(
                    self.model, accepted_mark_distilled_visual_event_adapter
                )
            )
            print(
                "Loaded accepted-mark-distilled visual-event adapter: "
                f"{accepted_mark_distilled_visual_event_adapter} "
                f"(arm={self.accepted_mark_distilled_visual_event_pilot['checkpoint']['config']['arm']})"
            )
        elif geometry_coupled_visual_event_adapter:
            from .geometry_coupled_visual_event_reentry_pbd import (
                load_geometry_coupled_visual_event_adapter,
            )

            self.geometry_coupled_visual_event_pilot = (
                load_geometry_coupled_visual_event_adapter(
                    self.model, geometry_coupled_visual_event_adapter
                )
            )
            print(
                "Loaded geometry-coupled visual-event adapter: "
                f"{geometry_coupled_visual_event_adapter} "
                f"(arm={self.geometry_coupled_visual_event_pilot['checkpoint']['config']['arm']})"
            )
        elif exchangeable_visual_event_adapter:
            from .exchangeable_visual_event_reentry_pbd import (
                load_exchangeable_visual_event_adapter,
            )

            self.exchangeable_visual_event_pilot = (
                load_exchangeable_visual_event_adapter(
                    self.model, exchangeable_visual_event_adapter
                )
            )
            print(
                "Loaded exchangeable visual-event adapter: "
                f"{exchangeable_visual_event_adapter} "
                f"(arm={self.exchangeable_visual_event_pilot['checkpoint']['config']['arm']})"
            )
        elif posterior_distilled_visual_event_adapter:
            from .posterior_distilled_event_reentry_pbd import (
                load_posterior_distilled_visual_event_adapter,
            )

            self.posterior_distilled_visual_event_pilot = (
                load_posterior_distilled_visual_event_adapter(
                    self.model, posterior_distilled_visual_event_adapter
                )
            )
            print(
                "Loaded posterior-distilled visual-event adapter: "
                f"{posterior_distilled_visual_event_adapter} "
                f"(arm={self.posterior_distilled_visual_event_pilot['checkpoint']['config']['arm']})"
            )
        elif posterior_guided_visual_event_adapter:
            from .posterior_guided_visual_event_pbd import (
                load_posterior_guided_visual_event_adapter,
            )

            self.posterior_guided_visual_event_pilot = (
                load_posterior_guided_visual_event_adapter(
                    self.model, posterior_guided_visual_event_adapter
                )
            )
            print(
                "Loaded posterior-guided visual-event adapter: "
                f"{posterior_guided_visual_event_adapter} "
                f"(arm={self.posterior_guided_visual_event_pilot['checkpoint']['config']['arm']})"
            )
        elif intensity_guided_visual_event_adapter:
            from .intensity_event_reentry_pbd import (
                load_intensity_guided_visual_event_adapter,
            )

            self.intensity_guided_visual_event_pilot = (
                load_intensity_guided_visual_event_adapter(
                    self.model, intensity_guided_visual_event_adapter
                )
            )
            print(
                "Loaded intensity-guided visual-event adapter: "
                f"{intensity_guided_visual_event_adapter} "
                f"(arm={self.intensity_guided_visual_event_pilot['checkpoint']['config']['arm']})"
            )

    @classmethod
    def from_args(cls, model_path: str, device: str, dtype: str) -> "LocateAnythingWorker":
        return cls(model_path=model_path, device=device, dtype=dtype)

    def predict(
        self,
        image: Any,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
        verbose: bool = False,
        do_sample: bool = True,
        top_p: float = 0.9,
        repetition_penalty: Optional[float] = 1.1,
        decode_mode: str = "baseline",
        return_debug_info: bool = False,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        assistant_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        torch = self.torch
        image_items = list(image) if isinstance(image, (list, tuple)) else [image]
        if not image_items:
            raise ValueError("LocateAnything inference requires at least one image")
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": item} for item in image_items),
                    {"type": "text", "text": question},
                ],
            }
        ]
        with torch.no_grad():
            text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = self.processor.process_vision_info(messages)
            inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
            assistant_prefix_tokens = 0
            if assistant_prefix:
                prefix_ids = self.tokenizer(
                    assistant_prefix,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].to(inputs["input_ids"].device)
                assistant_prefix_tokens = int(prefix_ids.shape[1])
                input_length = int(inputs["input_ids"].shape[1]) + assistant_prefix_tokens
                model_max_length = int(self.tokenizer.model_max_length)
                if input_length >= model_max_length:
                    raise ValueError(
                        "Assistant-prefix intervention exceeds the model context: "
                        f"{input_length} >= {model_max_length}; refusing to truncate"
                    )
                inputs["input_ids"] = torch.cat(
                    (inputs["input_ids"], prefix_ids), dim=1
                )
                inputs["attention_mask"] = torch.cat(
                    (
                        inputs["attention_mask"],
                        torch.ones_like(prefix_ids, dtype=inputs["attention_mask"].dtype),
                    ),
                    dim=1,
                )
            input_tokens = int(inputs["input_ids"].shape[1])
            generate_kwargs: Dict[str, Any] = {
                "pixel_values": inputs["pixel_values"].to(self.dtype),
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "image_grid_hws": inputs.get("image_grid_hws", None),
                "tokenizer": self.tokenizer,
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
                "generation_mode": generation_mode,
                "decode_mode": decode_mode,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "verbose": verbose,
                "return_debug_info": return_debug_info,
            }
            if repetition_penalty is not None:
                generate_kwargs["repetition_penalty"] = repetition_penalty
            if generation_kwargs:
                generate_kwargs.update(generation_kwargs)
            active_pilot = (
                self.pointrelay_pilot
                or self.evidence_pilot
                or self.witness_pilot
                or self.adaptive_memory_pilot
                or self.dense_address_pilot
                or self.marked_residual_pilot
                or self.latent_address_pilot
                or self.distribution_address_pilot
                or self.residual_set_pilot
                or self.residual_geometry_pilot
                or self.evidence_sentinel_pilot
                or self.latent_point_process_pilot
                or self.posterior_guided_visual_event_pilot
                or self.posterior_distilled_visual_event_pilot
                or self.exchangeable_visual_event_pilot
                or self.geometry_coupled_visual_event_pilot
                or self.accepted_mark_distilled_visual_event_pilot
                or self.mass_conserving_visual_event_pilot
                or self.open_ended_residual_event_pilot
                or self.point_marked_residual_event_pilot
                or self.intensity_guided_visual_event_pilot
                or self.semantic_aligned_event_pilot
                or self.ravel_pilot
            )
            if active_pilot is not None:
                controller = active_pilot.get("controller")
                if controller is not None and hasattr(controller, "set_inference_task"):
                    controller.set_inference_task(question, tokenizer=self.tokenizer)
                if controller is not None and hasattr(controller, "set_inference_image_layout"):
                    image_grid_hws = inputs.get("image_grid_hws", None)
                    if image_grid_hws is None:
                        raise RuntimeError("Localization pilot inference requires image_grid_hws")
                    controller.set_inference_image_layout(inputs["input_ids"], image_grid_hws)
                if controller is not None and hasattr(controller, "prepare_pixel_field"):
                    image_grid_hws = inputs.get("image_grid_hws", None)
                    if image_grid_hws is None:
                        raise RuntimeError("Dense address inference requires image_grid_hws")
                    controller.prepare_pixel_field(
                        inputs["pixel_values"].to(self.dtype), image_grid_hws
                    )
                if controller is not None and hasattr(controller, "prepare_inference_memory"):
                    image_grid_hws = inputs.get("image_grid_hws", None)
                    if image_grid_hws is None:
                        raise RuntimeError("Adaptive memory inference requires image_grid_hws")
                    controller.prepare_inference_memory(
                        self.model,
                        inputs["pixel_values"].to(self.dtype),
                        image_grid_hws,
                    )
            model_module = None
            original_handle_pattern = None
            if active_pilot is not None:
                controller = active_pilot.get("controller")
                if (
                    controller is not None
                    and hasattr(controller, "observe_generated_block")
                    and not bool(
                        getattr(controller, "integrated_generation_observer", False)
                    )
                ):
                    model_module = sys.modules.get(self.model.__class__.__module__)
                    original_handle_pattern = getattr(model_module, "handle_pattern", None)
                    if original_handle_pattern is None:
                        raise RuntimeError("LocateAnything generation module does not expose handle_pattern")

                    def observed_handle_pattern(*args: Any, **kwargs: Any) -> Any:
                        result = original_handle_pattern(*args, **kwargs)
                        if isinstance(result, dict):
                            controller.observe_generated_block(
                                str(result.get("type", "")),
                                result.get("tokens", ()),
                            )
                        return result

                    model_module.handle_pattern = observed_handle_pattern
            try:
                response = self.model.generate(
                    **generate_kwargs,
                )
            finally:
                if model_module is not None and original_handle_pattern is not None:
                    model_module.handle_pattern = original_handle_pattern
        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        if active_pilot is not None:
            controller = active_pilot.get("controller")
            if controller is not None:
                diagnostics = dict(getattr(controller, "last_diagnostics", {}) or {})
                coverage = getattr(controller, "_coverage", None)
                if coverage is not None:
                    diagnostics["memory_coverage_sum"] = float(
                        coverage.detach().float().sum().cpu()
                    )
                if hasattr(controller, "route_call_count"):
                    diagnostics["route_call_count"] = int(controller.route_call_count)
                result["pilot_diagnostics"] = diagnostics
        result["input_tokens"] = input_tokens
        result["assistant_prefix_tokens"] = assistant_prefix_tokens
        return result

    def run_sample(
        self,
        sample: EvalSample,
        image_path: str | Path,
        generation_mode: str = "hybrid",
        decode_mode: str = "baseline",
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> LocateAnythingResult:
        from PIL import Image

        torch = self.torch
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        question = locateanything_prompt(sample)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = self.predict(
            image,
            question,
            generation_mode=generation_mode,
            decode_mode=decode_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            generation_kwargs=generation_kwargs,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        answer = str(output.get("answer", ""))
        stats = output.get("stats") or {}
        tokens_generated = stats.get("tokens_generated") if isinstance(stats, dict) else None
        gpu_memory_mb = None
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        return LocateAnythingResult(
            answer=answer,
            pred_hboxes=parse_locateanything_boxes(answer, width, height, sample.target_class),
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
            gpu_memory_mb=gpu_memory_mb,
            raw_stats=stats if isinstance(stats, dict) else {},
        )


def resolve_image_path(sample: EvalSample, dataset_roots: Dict[str, str], extensions: Iterable[str] = (".jpg", ".png", ".tif", ".tiff")) -> Path:
    candidates: List[Path] = []
    if sample.source:
        source = Path(sample.source)
        candidates.append(source)
        if not source.is_absolute():
            root = dataset_roots.get(sample.dataset)
            if root:
                candidates.append(Path(root) / source)
    root = dataset_roots.get(sample.dataset)
    if root:
        for suffix in extensions:
            candidates.append(Path(root) / f"{sample.image_id}{suffix}")
        candidates.append(Path(root) / sample.image_id)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image file found for sample {sample.dataset}/{sample.image_id}; tried {candidates}")
