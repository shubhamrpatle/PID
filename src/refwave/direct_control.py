"""Class-isolated direct-box control using the existing episode annotations."""

import re


def detection_question(reference):
    return f"Locate all the instances that matches the following description: {reference}."


def isolated_conversations(row):
    """Partition the existing answer, without sampling classes or dropping targets."""
    answer = row["direct"]["conversations"][1]["value"]
    groups = list(re.finditer(r"<ref>(.*?)</ref>(.*?)(?=<ref>|\Z)", answer, re.DOTALL))
    if "".join(m.group(0) for m in groups) != answer:
        raise ValueError("Class-isolated detection requires grouped reference answers")
    references = [b["reference"] for b in row["round1"]["branches"]]
    if [m.group(1) for m in groups] != references:
        raise ValueError("Direct answer groups must match the queried references in order")
    return [
        [
            {"from": "human", "value": "<image>\n" + detection_question(reference)},
            {"from": "gpt", "value": match.group(0)},
        ]
        for reference, match in zip(references, groups)
    ]
