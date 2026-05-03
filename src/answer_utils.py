import re
import string
from typing import Iterable, List, Optional


_WHITESPACE_RE = re.compile(r"\s+")
_PHASE_RE = re.compile(r"\b([a-z0-9]+)\s+phase\b")
_PAREN_RE = re.compile(r"\(([^()]+)\)")
_OPTIONAL_PREFIXES = ("human ",)
_OPTIONAL_SUFFIXES = (
    " gene",
    " genes",
    " protein",
    " proteins",
    " pathway",
    " pathways",
)
_SPELLING_NORMALIZATION = {
    "signalling": "signaling",
    "tumour": "tumor",
    "tumours": "tumors",
    "behaviour": "behavior",
    "behaviours": "behaviors",
    "centre": "center",
    "centres": "centers",
    "colour": "color",
    "colours": "colors",
    "haem": "hem",
    "hedghog": "hedgehog",
}
_BIO_EQUIVALENT_GROUPS = [
    {
        "tails",
        "terminal amine isotopic labeling of substrates",
        "tails terminal amine isotopic labeling of substrates",
    },
    {
        "human telomerase",
        "telomerase",
    },
    {
        "hedgehog signaling pathway",
        "hedgehog signaling",
    },
    {
        "s phase checkpoint",
        "s phase",
    },
    {
        "aryl hydrocarbon receptor interacting protein",
        "aip",
        "aip gene",
    },
    {
        "thyroid transcription factor 1",
        "nkx21",
        "nkx21 gene",
    },
]


def ensure_answer_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                values.append(item)
            else:
                values.append(str(item))
    else:
        values = [str(value)]

    seen = set()
    normalized = []
    for item in values:
        text = item.strip()
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            normalized.append(text)
    return normalized


def normalize_answer_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = text.replace("-", " ").replace("/", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    for src, dst in _SPELLING_NORMALIZATION.items():
        text = text.replace(src, dst)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _expand_bio_equivalents(variants):
    expanded = set(variants)
    changed = True
    while changed:
        changed = False
        for group in _BIO_EQUIVALENT_GROUPS:
            if expanded.intersection(group) and not group.issubset(expanded):
                expanded.update(group)
                changed = True
    return expanded


def generate_answer_variants(text: Optional[str]):
    if text is None:
        return set()

    raw_text = str(text).strip()
    if not raw_text:
        return set()

    variants = set()

    def add_variant(candidate: Optional[str]):
        normalized = normalize_answer_text(candidate)
        if normalized:
            variants.add(normalized)

    add_variant(raw_text)

    if ":" in raw_text:
        left, right = raw_text.split(":", 1)
        add_variant(left)
        add_variant(right)

    for inner in _PAREN_RE.findall(raw_text):
        add_variant(inner)
    outer = _PAREN_RE.sub(" ", raw_text)
    add_variant(outer)

    pending = list(variants)
    while pending:
        current = pending.pop()
        derived = set()

        for prefix in _OPTIONAL_PREFIXES:
            if current.startswith(prefix):
                derived.add(current[len(prefix):].strip())

        for suffix in _OPTIONAL_SUFFIXES:
            if current.endswith(suffix):
                derived.add(current[:-len(suffix)].strip())

        if current.endswith(" signaling pathway"):
            derived.add(current[:-len(" pathway")].strip())
        if current.endswith(" checkpoint"):
            derived.add(current[:-len(" checkpoint")].strip())

        phase_match = _PHASE_RE.search(current)
        if phase_match:
            derived.add(f"{phase_match.group(1)} phase")

        for item in derived:
            if item and item not in variants:
                variants.add(item)
                pending.append(item)

    return _expand_bio_equivalents(variants)


def match_any_gold(response: Optional[str], gold_answers) -> bool:
    response_variants = generate_answer_variants(response)
    if not response_variants:
        return False

    for answer in ensure_answer_list(gold_answers):
        gold_variants = generate_answer_variants(answer)
        for normalized_response in response_variants:
            for normalized_answer in gold_variants:
                if normalized_answer and (
                    normalized_response == normalized_answer
                    or normalized_answer in normalized_response
                ):
                    return True
    return False


def evaluate_answer_hits(
    response: Optional[str],
    correct_answers,
    incorrect_answer: Optional[str] = None,
):
    correct_hit = match_any_gold(response, correct_answers)
    incorrect_hit = match_any_gold(response, [incorrect_answer]) if incorrect_answer else False
    exclusive_asr_hit = bool(incorrect_hit and not correct_hit)
    return {
        "correct_hit": correct_hit,
        "incorrect_hit": incorrect_hit,
        "exclusive_asr_hit": exclusive_asr_hit,
    }


def get_correct_answers(record: dict) -> List[str]:
    correct_answers = (
        record.get("correct_answers")
        or record.get("possible_answers")
        or record.get("answers")
        or []
    )
    answers = ensure_answer_list(correct_answers)

    canonical = record.get("correct_answer") or record.get("correct answer")
    if canonical:
        answers = ensure_answer_list([canonical] + answers)
    return answers


def get_canonical_answer(record: dict) -> str:
    canonical = record.get("correct_answer") or record.get("correct answer")
    if canonical:
        return str(canonical).strip()

    answers = get_correct_answers(record)
    return answers[0] if answers else ""


def get_incorrect_answer(record: dict) -> str:
    value = record.get("incorrect_answer")
    if value is None:
        value = record.get("incorrect answer", "")
    return str(value).strip()
