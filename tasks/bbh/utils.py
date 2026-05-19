import re


def doc_to_text(doc):
    return (
        f"{doc['input']}\n"
        f"Please reason step by step, and put your final answer"
        f"{_answer_hint(doc['target'])} within \\boxed{{}}."
    )


def _answer_hint(target):
    if re.fullmatch(r"\([A-Z]\)", target):
        return " (your chosen option, e.g., (A), (B), (C), etc.)"
    if target in {"True", "False"}:
        return " ('True' or 'False')"
    if target in {"Yes", "No"}:
        return " ('Yes' or 'No')"
    if target in {"yes", "no"}:
        return " ('yes' or 'no')"
    if target in {"valid", "invalid"}:
        return " ('valid' or 'invalid')"
    return ""


def _last_match(text, patterns):
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    return None


def _extract_answer(text, target):
    matches = re.findall(r"boxed\{(.*?)\}", text)
    if matches:
        text = matches[-1].strip()

    if re.fullmatch(r"\([A-Z]\)", target):
        answer = _last_match(text, [r"\(([A-Z])\)", r"\b([A-Z])\b"])
        return f"({answer})" if answer else text
    if target in {"True", "False"}:
        answer = _last_match(text, [r"\b(True|False)\b", r"\b(true|false)\b"])
        return answer.title() if answer else text
    if target in {"Yes", "No"}:
        answer = _last_match(text, [r"\b(Yes|No)\b", r"\b(yes|no)\b"])
        return answer.title() if answer else text
    if target in {"yes", "no"}:
        answer = _last_match(text, [r"\b(yes|no)\b", r"\b(Yes|No)\b"])
        return answer.lower() if answer else text
    if target in {"valid", "invalid"}:
        answer = _last_match(text, [r"\b(valid|invalid)\b", r"\b(Valid|Invalid)\b"])
        return answer.lower() if answer else text
    if re.fullmatch(r"-?\d+", target):
        answer = _last_match(text, [r"\b-?\d+\b"])
        return answer if answer else text
    return text.strip()


def process_results(doc, results):
    pred = _extract_answer(results[0], doc["target"])
    return {"exact_match": float(pred == doc["target"])}
