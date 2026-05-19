import re


def doc_to_text(doc):
    choices = {
        label: text
        for label, text in zip(doc["choices"]["label"], doc["choices"]["text"])
    }
    options = "\n".join(f"({label}) {text}" for label, text in choices.items())
    return (
        "Please reason step by step, and put your final answer "
        "(your chosen option, e.g., (A), (B), (C), etc.) within \\boxed{}.\n\n"
        f"{doc['question']}\nOptions:\n{options}"
    )


def doc_to_target(doc):
    return doc["answerKey"]


def _extract_choice(text):
    matches = re.findall(r"boxed\{(.*?)\}", text)
    if matches:
        text = matches[-1].strip()
    for pattern in (r"\(([A-D])\)", r"\b([A-D])\b"):
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    return text.strip()


def process_results(doc, results):
    pred = _extract_choice(results[0])
    return {"exact_match": float(pred == doc_to_target(doc))}
