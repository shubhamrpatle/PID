"""Compile only the routes used by the selected cluster experiment."""
from .unified_episode import (
    SCHEMA, UnifiedEpisodeDataset, round1_conversation, round2_conversation, split_waves,
)


class _NativeMTP6:
    """The pinned Eagle helper takes block_size at construction, not per call."""
    def __init__(self, helper):
        self.helper = helper

    def get_targets_flag_with_mtp(self, input_ids, block_size=6):
        if block_size != 6 or self.helper.block_size != 6:
            raise ValueError("RefWave HBB uses native PBD6")
        return self.helper.get_targets_flag_with_mtp(input_ids)


class ClusterDataset(UnifiedEpisodeDataset):
    def __init__(self, *args, method="refwave", **kwargs):
        super().__init__(*args, **kwargs)
        self.mtp = _NativeMTP6(self.mtp)
        self.method = method

    def __getitem__(self, index):
        row = self.rows[index]
        if row.get("schema_version") != SCHEMA or row["task"] not in ("detection", "grounding"):
            raise ValueError("Expected a RefWave detection/grounding episode")
        image = row["image"]
        pixels, grids, image_tokens = self._prepare_image(image, row["direct"]["conversations"])
        direct, round1, round2 = [], [], []
        if self.method == "standard":
            direct.append(self._compile(image, row["direct"]["conversations"], image_tokens,
                route="direct", branch_id=row["episode_id"] + ":direct", polarity="positive"))
        else:
            for branch in row["round1"]["branches"]:
                round1.append(self._compile(image, round1_conversation(branch, "geometry_only"),
                    image_tokens, route="round1", branch_id=branch["branch_id"], polarity=branch["polarity"]))
            for branch in row["round2"]["branches"]:
                round2.append(self._compile(image, round2_conversation(branch), image_tokens,
                    route="round2", branch_id=branch["branch_id"], polarity=branch["polarity"],
                    negative_type=branch.get("negative_type")))
        def waves(branches):
            return split_waves(branches, max_sequence=self.max_sequence,
                               maximum_wave_size=self.maximum_wave_size) if branches else []
        return dict(episode_id=row["episode_id"], task=row["task"], stage=row["stage"],
            dataset=row["dataset"], pixel_values=pixels, image_grid_hws=grids, image_tokens=image_tokens,
            direct_waves=waves(direct), round1_waves=waves(round1), round2_waves=waves(round2),
            episode_waves=waves(direct + round1 + round2))
