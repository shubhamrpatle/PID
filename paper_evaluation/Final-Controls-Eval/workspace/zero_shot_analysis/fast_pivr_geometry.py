"""Opt-in box-only decoder: batched postprocessing and exact LM-head row selection."""

import torch
import torch.nn.functional as F

from .pixel_pivr_wave import PixelPIVRWaveDecoder
from .pointrelay_pbd import base_causal_lm


class FastPIVRGeometryDecoder(PixelPIVRWaveDecoder):
    def __init__(self, *args, compact_head=False, **kwargs):
        super().__init__(*args, **kwargs)
        if self.geometry_prefix_mode != "box_only":
            raise ValueError("Fast geometry requires the existing box_only grammar")
        self.compact_head = bool(compact_head)
        self._head_rows = None

    @torch.no_grad()
    def _prepared_global_suffixes(self, prepared_image, branches):
        if self.visual_context != "global_only" or self.prefix_cache_mode != "shared":
            raise ValueError("Fast suffix tokenization requires shared global-only decoding")
        if prepared_image.model is not self.model or prepared_image.processor is not self.processor:
            raise ValueError("Cannot reuse another model/processor's visual prefix")
        projected = prepared_image.projected_global
        placeholder = ("<image 1>" + self.processor.image_start_token
            + self.processor.image_token * int(projected.shape[0]) + self.processor.image_end_token)
        leading, tails = None, []
        for branch in branches:
            text = self.processor.py_apply_chat_template(self._messages(prepared_image.image, branch),
                tokenize=False, add_generation_prompt=True)
            if text.count("<image-1>") != 1:
                raise RuntimeError("Unexpected global-image placeholder layout")
            head, tail = text.split("<image-1>")
            if leading is not None and leading != head:
                raise RuntimeError("Branch text before the image changed")
            leading = head
            tails.append(tail)
        # </img> is an atomic special token: tokenize its prefix once, then only
        # the short textual suffixes. No 1390-token image prefix per address.
        prefix = self.tokenizer(leading + placeholder, add_special_tokens=True)["input_ids"]
        if prefix != prepared_image.prefix or prefix[-1] != self.image_end_id:
            raise RuntimeError("Round-2 image prefix differs from cached Round 1")
        if self.image_end_id not in self.tokenizer.all_special_ids:
            raise RuntimeError("Suffix-only tokenization requires an atomic image-end boundary")
        suffixes = self.tokenizer(tails, add_special_tokens=False)["input_ids"]
        empty = projected.new_empty((0, int(projected.shape[-1])))
        return (torch.tensor(prefix, dtype=torch.long),
                [torch.tensor(s, dtype=torch.long) for s in suffixes], [empty for _ in branches])

    def _geometry_from_hidden(self, hidden, branches, *, allow_none):
        ids = self.model.token_ids
        start, end = ids["coord_start_token_id"], ids["coord_end_token_id"]
        weight = base_causal_lm(self.model.language_model).lm_head.weight
        if self.compact_head:
            # Inference-only decoder is bound to one immutable checkpoint.
            if self._head_rows is None:
                self._head_rows = torch.cat((weight[start:end + 1],
                    weight[ids["none_token_id"]:ids["none_token_id"] + 1])).detach()
            logits = F.linear(hidden, self._head_rows)
            coordinates = logits[:, 1:5, :-1].float()
            none = logits[:, 1, -1].float()
        else:
            logits = F.linear(hidden, weight)
            coordinates = logits[:, 1:5, start:end + 1].float()
            none = logits[:, 1, ids["none_token_id"]].float()
        return self._decode_coordinates(coordinates, none, branches, allow_none=allow_none)

    def _decode_coordinates(self, coordinates, none, branches, *, allow_none):
        ids = self.model.token_ids
        values, indices = torch.topk(coordinates, k=5, dim=-1)
        coords = torch.round((indices.float() * values.softmax(dim=-1)).sum(dim=-1)).long().clamp(0, 1000)
        x = torch.sort(coords[:, (0, 2)], dim=-1).values
        y = torch.sort(coords[:, (1, 3)], dim=-1).values
        coords = torch.stack((x[:, 0], y[:, 0], x[:, 1], y[:, 1]), dim=-1)
        empty = none >= values[:, 0, 0] if allow_none else torch.zeros(len(branches), device=coords.device, dtype=torch.bool)
        # Transfer once per wave, not four scalar synchronizations per address.
        packed = torch.cat((empty.long().unsqueeze(-1), coords), dim=-1).cpu().tolist()
        results = []
        for branch, (is_empty, *xyxy) in zip(branches, packed):
            tokens = ([ids["box_start_token_id"], ids["none_token_id"], ids["box_end_token_id"]]
                      if is_empty else [ids["box_start_token_id"],
                          *(ids["coord_start_token_id"] + v for v in xyxy), ids["box_end_token_id"]])
            block = self.tokenizer.decode(tokens, skip_special_tokens=False)
            results.append(dict(address_id=branch.address_id, label=branch.label,
                block_type="empty_box" if is_empty else "coord_box", block_tokens=tokens,
                answer=f"<ref>{branch.label}</ref>{block}"))
        return results
