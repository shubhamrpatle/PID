# Third-Party Source

`src/refwave/native_mtp.py` extracts the `get_targets_flag_with_mtp` method from
NVlabs/Eagle, commit `8442db3b79f7fd2357e468e6eecdd9b6a82049ff`, file
`Embodied/eaglevl/train/locany_finetune_magi_stream.py`.
The label-generation algorithm is unchanged; its enclosing dataset constructor
is replaced by a minimal processor-only constructor. Source:
https://github.com/NVlabs/Eagle/tree/8442db3b79f7fd2357e468e6eecdd9b6a82049ff

The upstream Apache-2.0 license is reproduced in `licenses/Eagle-APACHE-2.0.txt`.
`messages.py` specializes the upstream message format for one image and a single
human/assistant turn; it rejects unsupported inputs rather than importing the
video, augmentation and trainer stack. `provenance.json` records source hashes.

The local RefWave core and wave decoders were extracted from the project's
Pixel-PIVR repository, with dependency isolation and the changes documented in
README.md. No original datasets or model weights are included in this repository.
The released LocateAnything checkpoint and each source dataset retain their own
licenses and use conditions. Downloading them does not change those conditions.
