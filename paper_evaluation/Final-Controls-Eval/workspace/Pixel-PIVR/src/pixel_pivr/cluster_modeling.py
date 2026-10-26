"""Batched RefWave episodes and Qwen LoRA + full output-head adaptation."""
from __future__ import annotations

from collections import Counter
import math
import torch
from torch.utils.checkpoint import checkpoint

from .lora import base_causal_lm, configure_lora_only
from .unified_episode import first_supervised, x0_length, build_branch_isolation_mask
from .unified_modeling import UnifiedEpisodeTrainingModel, branch_objective_weights, objective_waves


def configure_lora_head(model, rank=32):
    audit = configure_lora_only(model, rank)
    causal = base_causal_lm(model.language_model)
    head = causal.lm_head
    if head.weight is causal.get_input_embeddings().weight:
        head.weight = torch.nn.Parameter(head.weight.detach().clone())
        causal.config.tie_word_embeddings = False
    head.requires_grad_(True)
    # Keep Adam's parameters/moments in FP32; forwards still use BF16 autocast.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    audit.update(head_parameters=sum(p.numel() for p in head.parameters()),
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 scope="Qwen LoRA + complete LM head; frozen input embeddings, vision and projector")
    return audit


def update_indices(order, cursor, per_gpu, world_size, rank):
    end = min(len(order), cursor + per_gpu * world_size)
    return order[cursor + rank:end:world_size], end


def wave_shape(wave):
    # Vector comparisons avoid a Python scalar loop over every image token.
    prefix = min(min(first_supervised(b["labels"]), x0_length(b["position_ids"])) for b in wave)
    first = wave[0]["input_ids"]
    for branch in wave[1:]:
        unequal = torch.nonzero(first[:prefix].ne(branch["input_ids"][:prefix]), as_tuple=False)
        if unequal.numel():
            prefix = int(unequal[0].item())
    return prefix + sum(b["input_ids"].numel()-prefix for b in wave), prefix


def pack_cluster_wave(wave, image_token_index, device, dtype):
    _, prefix = wave_shape(wave)
    values = {key: torch.cat([wave[0][key][:prefix], *[b[key][prefix:] for b in wave]]).unsqueeze(0).to(device)
              for key in ("input_ids", "labels", "position_ids")}
    if values["labels"][:, :prefix].ne(-100).any() or not values["input_ids"].eq(image_token_index).any():
        raise ValueError("Invalid image/label shared prefix")
    mask, spans = build_branch_isolation_mask(wave, prefix, device=device, dtype=dtype)
    return dict(values, attention_mask=mask, branch_ranges=spans, prefix_length=prefix)


def group_jobs(jobs, max_batch, token_budget):
    """Bucket similar wave lengths. A large indivisible wave runs alone."""
    pending = sorted(jobs, key=lambda job: wave_shape(job[1])[0])
    group = []
    for job in pending:
        length = wave_shape(job[1])[0]
        if group and (len(group) == max_batch or (len(group) + 1) * length > token_budget):
            yield group
            group = []
        group.append(job)
    if group:
        yield group


def pack_batch(waves, image_token_index, device, dtype):
    items = [pack_cluster_wave(wave, image_token_index=image_token_index, device=device, dtype=dtype)
             for wave in waves]
    size = max(item["input_ids"].shape[1] for item in items)
    batch = len(items)
    ids = torch.zeros((batch, size), dtype=torch.long, device=device)
    positions = torch.zeros_like(ids)
    mask = torch.full((batch, 1, size, size), -torch.inf, dtype=dtype, device=device)
    for index, item in enumerate(items):
        n = item["input_ids"].shape[1]
        ids[index, :n] = item["input_ids"][0]
        positions[index, :n] = item["position_ids"][0]
        mask[index, :, :n, :n] = item["attention_mask"][0]
        # Dummy queries see one prefix token, never an all-masked softmax row.
        mask[index, :, n:, 0] = 0
    return ids, positions, mask, items


def branch_ce(states, labels, branch_range, head, chunk_size=64):
    start, stop = branch_range
    positions = torch.arange(start, stop, device=states.device)
    targets = labels[0, start:stop]
    valid = targets.ne(-100) & positions.gt(0)
    positions, targets = positions[valid], targets[valid]
    if not len(targets):
        raise ValueError("Branch contains no supervised tokens")
    total = states.new_zeros((), dtype=torch.float32)
    def loss_chunk(hidden, target, weight):
        return torch.nn.functional.cross_entropy(
            torch.nn.functional.linear(hidden, weight).float(), target, reduction="sum")
    for offset in range(0, len(targets), chunk_size):
        hidden = states[0, positions[offset:offset + chunk_size] - 1]
        target = targets[offset:offset + chunk_size]
        if torch.is_grad_enabled():
            total = total + checkpoint(loss_chunk, hidden, target, head, use_reentrant=False)
        else:
            total = total + loss_chunk(hidden, target, head)
    return total / len(targets), len(targets)


def group_cross_entropy(states, items, waves, weights, head, divisor=1.0, chunk_size=256):
    """Fuse supervised tokens across branches while preserving episode weights."""
    hidden, targets, multipliers, owners, counts = [], [], [], [], []
    for row, (item, wave, route_weights) in enumerate(zip(items, waves, weights)):
        for branch, (start, stop) in zip(wave, item["branch_ranges"]):
            positions = torch.arange(start, stop, device=states.device)
            labels = item["labels"][0, start:stop]
            valid = labels.ne(-100) & positions.gt(0)
            positions, labels = positions[valid], labels[valid]
            count = len(labels)
            if not count:
                raise ValueError("Branch contains no supervised tokens")
            hidden.append(states[row, positions-1])
            targets.append(labels)
            multipliers.append(torch.full((count,), route_weights[branch["branch_id"]]/count/divisor,
                                          device=states.device, dtype=torch.float32))
            owners.append(torch.full((count,), len(counts), device=states.device, dtype=torch.long))
            counts.append(count)
    hidden, targets = torch.cat(hidden), torch.cat(targets)
    multipliers, owners = torch.cat(multipliers), torch.cat(owners)
    sums = torch.zeros(len(counts), device=states.device, dtype=torch.float32)
    total = states.new_zeros((), dtype=torch.float32)
    def chunk_loss(hidden, target, weight, multiplier):
        loss = torch.nn.functional.cross_entropy(
            torch.nn.functional.linear(hidden, weight).float(), target, reduction="none")
        return (loss*multiplier).sum(), loss.detach()
    for start in range(0, len(targets), chunk_size):
        span = slice(start, start+chunk_size)
        inputs = (hidden[span], targets[span], head, multipliers[span])
        loss, detached = (checkpoint(chunk_loss, *inputs, use_reentrant=False)
                          if torch.is_grad_enabled() else chunk_loss(*inputs))
        total = total + loss
        sums.scatter_add_(0, owners[span], detached)
    means = (sums / torch.tensor(counts, device=states.device)).cpu().tolist()
    return total, list(zip(means, counts))


class ClusterEpisodeModel(UnifiedEpisodeTrainingModel):
    def batch(self, samples, *, divisor=1.0, backward=True, token_budget=8192):
        if not samples:
            return []
        projected = [self.encode_global(sample) for sample in samples]
        weights, jobs, rows = [], [], []
        for index, sample in enumerate(samples):
            by_route = {route: sample[route + "_waves"] for route in ("direct", "round1", "round2")}
            weights.append(branch_objective_weights(sample["task"], by_route, objective_mode=self.objective_mode))
            waves = objective_waves(sample, self.objective_mode)
            ids = [b["branch_id"] for wave in waves for b in wave]
            if len(ids) != len(set(ids)) or set(ids) != set(weights[-1]):
                raise ValueError("Episode branches differ from objective")
            jobs.extend((index, wave) for wave in waves)
            rows.append(dict(task=sample["task"], loss=0.0, route_sum=Counter(), route_weight=Counter(),
                             supervised_tokens=0, waves=len(waves)))
        for group in group_jobs(jobs, len(samples), token_budget):
            ids, positions, mask, items = pack_batch(
                [wave for _, wave in group], int(self.model.image_token_index), self.device, torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embeddings = self.model.language_model.get_input_embeddings()(ids).clone()
                for b, (sample_index, _) in enumerate(group):
                    n = items[b]["input_ids"].shape[1]
                    selected = ids[b, :n].eq(int(self.model.image_token_index))
                    if int(selected.sum()) != projected[sample_index].shape[0]:
                        raise ValueError("Visual token count mismatch")
                    embeddings[b, :n][selected] = projected[sample_index].to(embeddings.dtype)
                self.causal.model._caw_pivr_explicit_attention_mask = mask
                try:
                    states = self.causal.model(inputs_embeds=embeddings, position_ids=positions,
                        use_cache=False, return_dict=True).last_hidden_state
                finally:
                    self.causal.model._caw_pivr_explicit_attention_mask = None
                objective, summaries = group_cross_entropy(states, items, [wave for _, wave in group],
                    [weights[index] for index, _ in group], self.causal.lm_head.weight, divisor)
                summaries = iter(summaries)
                for b, (sample_index, wave) in enumerate(group):
                    for branch, span in zip(wave, items[b]["branch_ranges"]):
                        loss, count = next(summaries)
                        weight = weights[sample_index][branch["branch_id"]]
                        row = rows[sample_index]
                        value = loss * weight
                        row["loss"] += value
                        row["route_sum"][branch["route"]] += value
                        row["route_weight"][branch["route"]] += weight
                        row["supervised_tokens"] += count
            if not math.isfinite(float(objective.detach())):
                raise FloatingPointError("Non-finite batch loss")
            if backward:
                objective.backward()
            del objective, states, embeddings, mask, items
        for row in rows:
            for route in ("direct", "round1", "round2"):
                row[route + "_loss"] = row["route_sum"][route] / max(row["route_weight"][route], 1e-12)
            del row["route_sum"], row["route_weight"]
        return rows
