import re


def doc_to_text(doc):
    return (
        "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        f"{doc['question']}"
    )


def _extract_number(text):
    text = text.replace(",", "")
    match = re.search(r"###\s*(-?\d+)", text)
    if match:
        return round(float(match.group(1)))

    matches = re.findall(r"(?<!\d)-?\d+(?:\.\d+)?", text)
    if matches:
        return round(float(matches[-1]))
    return "No answer found"


def doc_to_target(doc):
    return _extract_number(doc["answer"])


def process_results(doc, results):
    pred = _extract_number(results[0])
    return {"exact_match": float(pred == doc_to_target(doc))}
