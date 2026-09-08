"""Single-image specialization of Eagle's multimodal conversation format."""

from pathlib import Path


def image_messages(image, conversations, data_root):
    if (
        len(conversations) != 2
        or conversations[0]["from"] != "human"
        or conversations[1]["from"] != "gpt"
    ):
        raise ValueError("Expected one human query followed by one target answer")
    prompt, answer = (entry["value"] for entry in conversations)
    if prompt.count("<image>") != 1 or "<image" in answer or "<video" in prompt:
        raise ValueError("RefWave records require exactly one <image> in the query")
    root = Path(data_root).resolve()
    path = (root / image).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Record image must be inside DATA_ROOT")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.replace("<image>", "<image-1>")},
                {"type": "image", "image": str(path)},
            ],
        },
        {"role": "assistant", "content": answer},
    ]
