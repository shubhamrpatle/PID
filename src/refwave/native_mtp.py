"""LA MTP6 labels extracted from NVlabs/Eagle; see THIRD_PARTY.md."""

from typing import Dict
import numpy as np
import torch

IGNORE_TOKEN_ID = -100


class NativeMTP6:

    def __init__(self, processor):
        self.processor = processor
        self.ds_name = "refwave"
        self.block_size = 6

    def get_targets_flag_with_mtp(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create MTP (Multi-Token Prediction) blocks with proper labels."""
        tokenizer = self.processor.tokenizer
        targets_flag = torch.zeros_like(input_ids)
        box_end_id = tokenizer.convert_tokens_to_ids("</box>")
        ref_end_id = tokenizer.convert_tokens_to_ids("</ref>")
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        null_id = tokenizer.convert_tokens_to_ids("<null>")
        mask_id = tokenizer.convert_tokens_to_ids("<text_mask>")
        start_header_idxs = torch.where(
            input_ids == tokenizer.convert_tokens_to_ids("<|im_start|>")
        )[0]
        assistant_idxs = torch.where(input_ids == tokenizer.convert_tokens_to_ids("assistant"))[0]
        eot_idxs = torch.where(input_ids == eos_id)[0]
        resp_start_position_ids = []
        resp_end_position_ids = []
        resp_label_ranges = []
        for assistant_idx in assistant_idxs:
            sets = list(set(start_header_idxs + 1))
            sets = [each.item() for each in sets]
            if assistant_idx.item() in sets:
                st = assistant_idx + 1
                for eot_idx in eot_idxs:
                    if eot_idx > st:
                        targets_flag[st + 1 : eot_idx + 1] = 1
                        resp_start_position_ids.append(st.item())
                        resp_end_position_ids.append((eot_idx + 1).item())
                        label_start = (st + 1).item()
                        label_end = eot_idx.item()
                        resp_label_ranges.append((label_start, label_end))
                        break
        targets = input_ids.clone()
        assert (
            targets_flag.sum() > 0
        ), f"No valid labels for training, skip sample in {self.ds_name}"
        targets[targets_flag == 0] = IGNORE_TOKEN_ID
        input_ids_np = input_ids.squeeze(0).cpu().numpy()
        targets_np = targets.squeeze(0).cpu().numpy()
        len_input_ids = len(input_ids_np)
        has_box = (input_ids == box_end_id).any().item()
        has_ref = (input_ids == ref_end_id).any().item()
        if not (has_box or has_ref):
            all_mask_input_ids = []
            all_mask_targets = []
            all_mask_positions = []
            for label_start, label_end in resp_label_ranges:
                resp_valid_positions = [
                    i for i in range(label_start, label_end + 1) if targets_np[i] != IGNORE_TOKEN_ID
                ]
                num_valid = len(resp_valid_positions)
                if num_valid == 0:
                    continue
                num_blocks = num_valid // self.block_size
                if num_blocks == 0:
                    continue
                used_tokens = num_blocks * self.block_size
                remaining = num_valid - used_tokens
                offset = np.random.randint(0, remaining + 1) if remaining > 0 else 0
                for block_idx in range(num_blocks):
                    start_pos = offset + block_idx * self.block_size
                    end_pos = start_pos + self.block_size
                    block_valid = resp_valid_positions[start_pos:end_pos]
                    if len(block_valid) == 0:
                        continue
                    first_token_idx = int(block_valid[0])
                    anchor_idx = max(first_token_idx - 1, 0)
                    target_block = np.full(self.block_size, IGNORE_TOKEN_ID, dtype=targets_np.dtype)
                    candidate_tokens = input_ids_np[block_valid]
                    for i, tok in enumerate(candidate_tokens):
                        if i >= self.block_size:
                            break
                        target_block[i] = tok
                        if tok == eos_id:
                            break
                    mask_input_block = np.full(self.block_size, mask_id, dtype=input_ids_np.dtype)
                    mask_input_block[0] = input_ids_np[anchor_idx]
                    pos_block = np.arange(anchor_idx, anchor_idx + self.block_size, dtype=np.int32)
                    all_mask_input_ids.append(mask_input_block)
                    all_mask_targets.append(target_block)
                    all_mask_positions.append(pos_block)
            if len(all_mask_input_ids) > 0:
                final_mask_ids = np.concatenate(all_mask_input_ids)
                final_mask_targets = np.concatenate(all_mask_targets)
                final_mask_positions = np.concatenate(all_mask_positions)
                pad_id = tokenizer.pad_token_id
                bridge_ignore = np.array([IGNORE_TOKEN_ID], dtype=targets_np.dtype)
                pad_token = np.array([pad_id], dtype=input_ids_np.dtype)
                input_ids_out = np.concatenate([input_ids_np, final_mask_ids, pad_token])
                targets_out = np.concatenate([targets_np, bridge_ignore, final_mask_targets])
                orig_pos = np.arange(len_input_ids, dtype=np.int32)
                pad_pos = np.array([final_mask_positions[-1] + 1], dtype=np.int32)
                position_ids_out = np.concatenate([orig_pos, final_mask_positions, pad_pos])
            else:
                input_ids_out = input_ids_np
                targets_out = targets_np
                position_ids_out = np.arange(len_input_ids, dtype=np.int32)
            input_ids = torch.tensor(input_ids_out, dtype=torch.long)
            targets = torch.tensor(targets_out, dtype=torch.long)
            position_ids = torch.tensor(position_ids_out, dtype=torch.long)
            return dict(
                input_ids=input_ids,
                labels=targets,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
                position_ids=position_ids,
            )
        all_mask_input_ids = []
        all_mask_targets = []
        all_mask_positions = []
        for start, end in zip(resp_start_position_ids, resp_end_position_ids):
            curr = start
            while curr < end:
                anchor_token = input_ids_np[curr]
                if anchor_token == eos_id:
                    break
                pred_start = curr + 1
                if pred_start > end:
                    break
                candidates = input_ids_np[pred_start : min(pred_start + self.block_size, end + 1)]
                if len(candidates) == 0:
                    break
                valid_len = len(candidates)
                eos_indices = np.where(candidates == eos_id)[0]
                if len(eos_indices) > 0:
                    first_eos_idx = eos_indices[0]
                    if first_eos_idx == 0:
                        valid_len = 1
                    else:
                        valid_len = first_eos_idx
                if valid_len > 1 or (len(eos_indices) > 0 and eos_indices[0] != 0):
                    ref_indices = np.where(candidates[:valid_len] == ref_end_id)[0]
                    if len(ref_indices) > 0:
                        valid_len = min(valid_len, ref_indices[0] + 1)
                    box_indices = np.where(candidates[:valid_len] == box_end_id)[0]
                    if len(box_indices) > 0:
                        valid_len = min(valid_len, box_indices[0] + 1)
                target_block = np.full(self.block_size, null_id, dtype=input_ids_np.dtype)
                target_block[:valid_len] = candidates[:valid_len]
                mask_input_block = np.full(self.block_size, mask_id, dtype=input_ids_np.dtype)
                mask_input_block[0] = anchor_token
                pos_block = np.arange(curr, curr + self.block_size, dtype=np.int32)
                all_mask_input_ids.append(mask_input_block)
                all_mask_targets.append(target_block)
                all_mask_positions.append(pos_block)
                curr += valid_len
        if len(all_mask_input_ids) > 0:
            final_mask_ids = np.concatenate(all_mask_input_ids)
            final_mask_targets = np.concatenate(all_mask_targets)
            final_mask_positions = np.concatenate(all_mask_positions)
            pad_id = tokenizer.pad_token_id
            bridge_ignore = np.array([IGNORE_TOKEN_ID], dtype=targets_np.dtype)
            pad_token = np.array([pad_id], dtype=input_ids_np.dtype)
            input_ids_out = np.concatenate([input_ids_np, final_mask_ids, pad_token])
            targets_out = np.concatenate([targets_np, bridge_ignore, final_mask_targets])
            orig_pos = np.arange(len_input_ids, dtype=np.int32)
            pad_pos = np.array([final_mask_positions[-1] + 1], dtype=np.int32)
            position_ids_out = np.concatenate([orig_pos, final_mask_positions, pad_pos])
        else:
            input_ids_out = input_ids_np
            targets_out = targets_np
            position_ids_out = np.arange(len_input_ids, dtype=np.int32)
        input_ids = torch.tensor(input_ids_out, dtype=torch.long)
        targets = torch.tensor(targets_out, dtype=torch.long)
        position_ids = torch.tensor(position_ids_out, dtype=torch.long)
        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            position_ids=position_ids,
        )
