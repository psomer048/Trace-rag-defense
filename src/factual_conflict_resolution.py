import json
import math
import re
from itertools import combinations
from typing import Dict, List, Optional

import networkx as nx

from .prompts_defense import (
    format_contradiction_type_classification_prompt,
    format_evidence_aware_judging_prompt,
    format_internal_knowledge_note_prompt,
    format_unit_slot_abstraction_prompt,
)


VALID_TYPED_LABELS = {"factual", "non-factual", "uncertain"}
VALID_NOTE_CONFIDENCE = {"low", "medium", "high"}
VALID_SUPPORT_NATURES = {
    "real_support",
    "self_consistent_narrative",
    "generic_background",
    "mixed",
    "unclear",
}
VALID_SUPPORT_PATTERNS = {
    "direct_slot_statement",
    "supported_relation_chain",
    "self_consistent_narrative",
    "generic_background",
    "mixed",
    "unclear",
}


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _extract_balanced_segment(text: str, start_idx: int) -> Optional[str]:
    if start_idx < 0 or start_idx >= len(text):
        return None

    opening = text[start_idx]
    if opening in "{[":
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for pos in range(start_idx, len(text)):
            ch = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return text[start_idx:pos + 1]
        return None

    if opening == '"':
        escaped = False
        for pos in range(start_idx + 1, len(text)):
            ch = text[pos]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                return text[start_idx:pos + 1]
        return None

    match = re.match(r"(true|false|null|-?\d+(?:\.\d+)?)", text[start_idx:], re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _extract_top_level_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    start_idx = text.find("{")
    if start_idx == -1:
        return None
    candidate = _extract_balanced_segment(text, start_idx)
    if not candidate:
        return None
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _clamp_score(value, default: float = 0.5) -> float:
    try:
        score = float(value)
    except Exception:
        score = default
    return max(0.0, min(1.0, score))


def _normalize_type_label(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "non factual": "non-factual",
        "non_factual": "non-factual",
        "nonfactual": "non-factual",
        "unclear": "uncertain",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in VALID_TYPED_LABELS else "uncertain"


def _normalize_confidence(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in VALID_NOTE_CONFIDENCE else "low"


def _normalize_support_nature(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "real": "real_support",
        "support": "real_support",
        "narrative": "self_consistent_narrative",
        "self_consistent": "self_consistent_narrative",
        "background": "generic_background",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in VALID_SUPPORT_NATURES else "unclear"


def _normalize_support_pattern(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "real_support": "supported_relation_chain",
        "relation_chain": "supported_relation_chain",
        "direct_answer": "direct_slot_statement",
        "slot_statement": "direct_slot_statement",
        "background": "generic_background",
        "narrative": "self_consistent_narrative",
        "self_consistent": "self_consistent_narrative",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in VALID_SUPPORT_PATTERNS else "unclear"


def _support_pattern_to_nature(value: str) -> str:
    pattern = _normalize_support_pattern(value)
    if pattern in {"direct_slot_statement", "supported_relation_chain"}:
        return "real_support"
    if pattern == "self_consistent_narrative":
        return "self_consistent_narrative"
    if pattern == "generic_background":
        return "generic_background"
    if pattern == "mixed":
        return "mixed"
    return "unclear"


def _normalize_claim_key(text: str) -> str:
    normalized = str(text or "").strip().lower()
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _truncate_text(text: str, max_chars: int = 400) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize_family_key(text: str) -> str:
    return _normalize_claim_key(text or "") or "generic::unknown"


def _extract_predicate_phrase(claim: str) -> str:
    text = str(claim or "").strip()
    if not text:
        return "unknown"

    lowered = text.lower()
    for marker in [" is an ", " is a ", " is ", " was an ", " was a ", " was "]:
        idx = lowered.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip(" .")
    return text.strip(" .")


def infer_query_form(query: str) -> str:
    normalized = str(query or "").strip().lower()
    if not normalized:
        return "direct_slot"

    explanatory_prefixes = (
        "why ",
        "how ",
        "how did ",
        "how does ",
        "how do ",
        "what caused ",
        "what led to ",
        "what explains ",
        "explain ",
    )
    if normalized.startswith(explanatory_prefixes):
        return "explanatory"

    direct_slot_keywords = (
        "occupation",
        "profession",
        "job",
        "work as",
        "career",
        "nationality",
        "citizenship",
        "title",
        "role",
        "position",
        "affiliation",
        "party",
        "capital",
        "population",
        "birthplace",
        "date of birth",
        "founded",
        "founder",
        "genre",
        "religion",
    )
    if any(keyword in normalized for keyword in direct_slot_keywords):
        return "direct_slot"

    if normalized.startswith(("who is ", "who was ", "what is ", "what was ")):
        return "identity_definition"

    relation_starters = (
        "who wrote ",
        "who founded ",
        "who directed ",
        "who composed ",
        "who discovered ",
        "which country ",
        "which city ",
        "which company ",
        "what year ",
        "when did ",
        "where did ",
        "where was ",
        "what is the capital of ",
        "who is the ",
        "who was the ",
    )
    if normalized.startswith(relation_starters):
        return "relation_lookup"

    return "direct_slot"


class SupportUnitConsolidator:
    def __init__(self, llm_model=None):
        self.llm_model = llm_model

    def _infer_query_slot(self, query: str) -> str:
        normalized = str(query or "").strip().lower()
        if any(keyword in normalized for keyword in ["occupation", "profession", "job", "work as", "career"]):
            return "occupation"
        return "generic"

    def _heuristic_abstraction(self, query: str, claim: str) -> Dict:
        slot = self._infer_query_slot(query)
        predicate = _extract_predicate_phrase(claim)
        base_answer = predicate.strip() or "unknown"

        if slot != "occupation":
            family_key = f"{slot}::{_normalize_family_key(base_answer)}"
            return {
                "base_answer": base_answer,
                "family_key": family_key,
                "abstraction_confidence": "low",
                "rationale": "Fallback heuristic kept the original claim granularity.",
            }

        lowered = predicate.lower()
        lowered = re.sub(r"\([^)]*\)", "", lowered)
        lowered = re.sub(r"\b(specializing in|specialised in|specialized in|who specializes in)\b.*", "", lowered).strip(" .")
        lowered = re.sub(r"\b(former|distinguished|renowned|licensed|career|professional|american|australian|british|english|canadian|dominican|dutch|french|german)\b", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()

        coordinated_patterns = [
            ("cartoonist", "illustrator", "cartoonist and illustrator"),
            ("photographer", "cinematographer", "photographer and cinematographer"),
            ("model", "influencer", "model and influencer"),
            ("musician", "composer", "musician and composer"),
            ("mathematician", "economist", "mathematician and economist"),
        ]
        for left, right, merged in coordinated_patterns:
            if left in lowered and right in lowered:
                return {
                    "base_answer": merged,
                    "family_key": f"{slot}::{_normalize_family_key(merged)}",
                    "abstraction_confidence": "medium",
                    "rationale": f"Fallback heuristic merged coordinated professions into '{merged}'.",
                }

        profession_heads = [
            "surgeon",
            "botanist",
            "politician",
            "actor",
            "actress",
            "illustrator",
            "cartoonist",
            "dentist",
            "geologist",
            "astronomer",
            "journalist",
            "mathematician",
            "economist",
            "businessman",
            "professor",
            "footballer",
            "photographer",
            "cinematographer",
            "musician",
            "composer",
            "producer",
            "filmmaker",
        ]
        for head in profession_heads:
            if re.search(rf"\b{re.escape(head)}s?\b", lowered):
                return {
                    "base_answer": head,
                    "family_key": f"{slot}::{_normalize_family_key(head)}",
                    "abstraction_confidence": "medium",
                    "rationale": f"Fallback heuristic abstracted the profession family to '{head}'.",
                }

        family_key = f"{slot}::{_normalize_family_key(base_answer)}"
        return {
            "base_answer": base_answer,
            "family_key": family_key,
            "abstraction_confidence": "low",
            "rationale": "Fallback heuristic kept the original occupation phrase.",
        }

    def _default_abstraction_entry(self, unit: Dict, query: str) -> Dict:
        fallback = self._heuristic_abstraction(query, unit.get("claim", "NONE"))
        return {
            "unit_id": unit["unit_id"],
            "base_answer": fallback["base_answer"],
            "family_key": fallback["family_key"],
            "abstraction_confidence": fallback["abstraction_confidence"],
            "rationale": fallback["rationale"],
        }

    def _parse_abstraction_response(self, response: str, query: str, canonical_units: List[Dict]) -> Dict[int, Dict]:
        cleaned = _strip_code_fences(response)
        payload = None
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = _extract_top_level_json(cleaned)

        results = {}
        if not payload:
            return results

        raw_units = payload.get("units", [])
        if not isinstance(raw_units, list):
            return results

        slot = self._infer_query_slot(query)
        unit_lookup = {item["unit_id"]: item for item in canonical_units}
        for item in raw_units:
            try:
                unit_id = int(item.get("unit_id"))
            except Exception:
                continue
            if unit_id not in unit_lookup:
                continue
            base_answer = str(item.get("base_answer", "")).strip() or self._default_abstraction_entry(unit_lookup[unit_id], query)["base_answer"]
            family_key = str(item.get("family_key", "")).strip()
            if not family_key:
                family_key = f"{slot}::{_normalize_family_key(base_answer)}"
            elif "::" not in family_key:
                family_key = f"{slot}::{_normalize_family_key(family_key)}"
            else:
                family_slot, family_value = family_key.split("::", 1)
                family_key = f"{family_slot.strip().lower()}::{_normalize_family_key(family_value)}"
            results[unit_id] = {
                "unit_id": unit_id,
                "base_answer": base_answer,
                "family_key": family_key,
                "abstraction_confidence": _normalize_confidence(item.get("abstraction_confidence")),
                "rationale": str(item.get("rationale", "")).strip() or "No abstraction rationale provided.",
            }
        return results

    def _abstract_units(
        self,
        query: str,
        documents: List[Dict],
        canonical_units: List[Dict],
        explanations_by_pair: Dict,
    ) -> Dict:
        abstractions = {}
        if self.llm_model and len(canonical_units) > 1:
            try:
                prompt_units = []
                for unit in canonical_units:
                    prompt_units.append(
                        {
                            "unit_id": unit["unit_id"],
                            "claim": unit.get("claim", "NONE"),
                            "doc_indices": [idx + 1 for idx in unit.get("active_doc_indices", [])],
                            "docs": self._build_unit_docs(documents, unit),
                        }
                    )
                response = self.llm_model.query(format_unit_slot_abstraction_prompt(query, prompt_units))
                abstractions.update(self._parse_abstraction_response(response, query, canonical_units))
            except Exception:
                abstractions = {}

        for unit in canonical_units:
            abstractions.setdefault(unit["unit_id"], self._default_abstraction_entry(unit, query))

        grouped = {}
        for unit in canonical_units:
            abstraction = abstractions[unit["unit_id"]]
            family_key = abstraction["family_key"]
            group = grouped.setdefault(
                family_key,
                {
                    "family_key": family_key,
                    "members": [],
                    "base_answer_counts": {},
                    "toxicity_scores": [],
                    "all_doc_indices": [],
                    "active_doc_indices": [],
                    "confidences": [],
                    "rationales": [],
                },
            )
            group["members"].append(unit)
            group["base_answer_counts"][abstraction["base_answer"]] = group["base_answer_counts"].get(abstraction["base_answer"], 0) + 1
            group["toxicity_scores"].append(unit.get("toxicity_score_max", 0.5))
            group["all_doc_indices"].extend(unit.get("all_doc_indices", []))
            group["active_doc_indices"].extend(unit.get("active_doc_indices", []))
            group["confidences"].append(abstraction.get("abstraction_confidence", "low"))
            if abstraction.get("rationale"):
                group["rationales"].append(abstraction["rationale"])

        abstracted_units = []
        canonical_to_abstracted = {}
        for abstracted_unit_id, group in enumerate(
            sorted(grouped.values(), key=lambda item: min(member["unit_id"] for member in item["members"]))
        ):
            members = sorted(group["members"], key=lambda item: item["unit_id"])
            active_doc_indices = sorted(set(group["active_doc_indices"]))
            all_doc_indices = sorted(set(group["all_doc_indices"]))
            gate_status = "rejected_by_toxicity" if not active_doc_indices else "survive"
            representative_claim = max(
                group["base_answer_counts"].items(),
                key=lambda item: (item[1], -len(_normalize_claim_key(item[0]))),
            )[0]
            for member in members:
                canonical_to_abstracted[member["unit_id"]] = abstracted_unit_id

            abstracted_units.append(
                {
                    "unit_id": abstracted_unit_id,
                    "claim": representative_claim,
                    "family_key": group["family_key"],
                    "member_unit_ids": [member["unit_id"] for member in members],
                    "member_claims": [member.get("claim", "NONE") for member in members],
                    "raw_unit_ids": sorted({raw_id for member in members for raw_id in member.get("raw_unit_ids", [])}),
                    "all_doc_indices": all_doc_indices,
                    "active_doc_indices": active_doc_indices,
                    "toxicity_score_max": max(group["toxicity_scores"]) if group["toxicity_scores"] else 0.5,
                    "gate_status": gate_status,
                    "abstraction_confidence": max(group["confidences"], key=lambda item: {"low": 0, "medium": 1, "high": 2}.get(item, 0)),
                    "abstraction_rationale": " | ".join(group["rationales"][:3]),
                }
            )

        active_units = [unit for unit in abstracted_units if unit.get("gate_status") != "rejected_by_toxicity"]
        pair_candidates = []
        pair_id = 1
        for unit1, unit2 in combinations(active_units, 2):
            if unit1.get("family_key") == unit2.get("family_key"):
                continue
            explanations = []
            for member1 in unit1.get("member_unit_ids", []):
                for member2 in unit2.get("member_unit_ids", []):
                    key = tuple(sorted((member1, member2)))
                    explanations.extend(item for item in explanations_by_pair.get(key, []) if item)
            pair_candidates.append(
                {
                    "pair_id": pair_id,
                    "unit1": unit1["unit_id"],
                    "unit2": unit2["unit_id"],
                    "unit1_claim": unit1.get("claim", "NONE"),
                    "unit2_claim": unit2.get("claim", "NONE"),
                    "unit1_docs": self._build_unit_docs(documents, unit1),
                    "unit2_docs": self._build_unit_docs(documents, unit2),
                    "cot_explanation": " | ".join(explanations) or "Abstracted answer families provide different candidate answers.",
                }
            )
            pair_id += 1

        return {
            "abstracted_units": abstracted_units,
            "canonical_to_abstracted": canonical_to_abstracted,
            "unit_abstractions": [abstractions[unit["unit_id"]] for unit in sorted(canonical_units, key=lambda item: item["unit_id"])],
            "pair_candidates": pair_candidates,
        }

    def consolidate_units(
        self,
        query: str,
        documents: List[Dict],
        cot_result: Dict,
        toxicity_step: Optional[Dict] = None,
    ) -> Dict:
        viewpoint_toxicity = {
            item.get("viewpoint_index"): item
            for item in (toxicity_step or {}).get("viewpoint_toxicity", [])
        }
        rejected_raw_units = set((toxicity_step or {}).get("rejected_viewpoints", []))

        grouped = {}
        for raw_unit in cot_result.get("support_units", []):
            raw_unit_id = int(raw_unit.get("unit_id", -1))
            claim = str(raw_unit.get("derived_answer", "")).strip() or "UNKNOWN"
            key = _normalize_claim_key(claim) or f"unit-{raw_unit_id}"
            group = grouped.setdefault(
                key,
                {
                    "normalized_claim": key,
                    "raw_units": [],
                    "claim_counts": {},
                    "all_doc_indices": [],
                    "active_doc_indices": [],
                    "toxicity_scores": [],
                },
            )

            toxicity_item = viewpoint_toxicity.get(raw_unit_id, {})
            toxicity_score = _clamp_score(toxicity_item.get("toxicity_score"), default=0.5)
            gate_status = "rejected_by_toxicity" if raw_unit_id in rejected_raw_units else "survive"
            doc_indices = []
            seen_doc_indices = set()
            for raw_doc_idx in raw_unit.get("docs", []):
                try:
                    doc_idx = int(raw_doc_idx)
                except Exception:
                    continue
                if doc_idx < 0 or doc_idx in seen_doc_indices:
                    continue
                seen_doc_indices.add(doc_idx)
                doc_indices.append(doc_idx)
            doc_indices.sort()

            group["raw_units"].append(
                {
                    "raw_unit_id": raw_unit_id,
                    "claim": claim,
                    "doc_indices": doc_indices,
                    "toxicity_score": toxicity_score,
                    "gate_status": gate_status,
                }
            )
            group["claim_counts"][claim] = group["claim_counts"].get(claim, 0) + 1
            group["toxicity_scores"].append(toxicity_score)
            group["all_doc_indices"].extend(doc_indices)
            if gate_status != "rejected_by_toxicity":
                group["active_doc_indices"].extend(doc_indices)

        canonical_units = []
        raw_to_canonical = {}
        for canonical_unit_id, group in enumerate(
            sorted(grouped.values(), key=lambda item: min(unit["raw_unit_id"] for unit in item["raw_units"]))
        ):
            active_doc_indices = sorted(set(group["active_doc_indices"]))
            all_doc_indices = sorted(set(group["all_doc_indices"]))
            raw_units = sorted(group["raw_units"], key=lambda item: item["raw_unit_id"])
            representative_raw = max(
                group["claim_counts"].items(),
                key=lambda item: (item[1], -len(_normalize_claim_key(item[0]))),
            )[0]
            gate_status = "rejected_by_toxicity" if not active_doc_indices else "survive"
            for raw_unit in raw_units:
                raw_to_canonical[raw_unit["raw_unit_id"]] = canonical_unit_id

            canonical_units.append(
                {
                    "unit_id": canonical_unit_id,
                    "claim": representative_raw,
                    "normalized_claim": group["normalized_claim"],
                    "raw_unit_ids": [item["raw_unit_id"] for item in raw_units],
                    "all_doc_indices": all_doc_indices,
                    "active_doc_indices": active_doc_indices,
                    "toxicity_score_max": max(group["toxicity_scores"]) if group["toxicity_scores"] else 0.5,
                    "gate_status": gate_status,
                }
            )

        explanations_by_pair = {}
        for conflict in cot_result.get("contradictory_units", []):
            unit1 = raw_to_canonical.get(conflict.get("unit1"))
            unit2 = raw_to_canonical.get(conflict.get("unit2"))
            if unit1 is None or unit2 is None or unit1 == unit2:
                continue
            key = tuple(sorted((unit1, unit2)))
            explanations_by_pair.setdefault(key, []).append(str(conflict.get("explanation", "")).strip())

        abstraction_step = self._abstract_units(query, documents, canonical_units, explanations_by_pair)
        abstracted_units = abstraction_step["abstracted_units"]

        return {
            "step": "support_unit_consolidation",
            "canonical_units": canonical_units,
            "abstracted_units": abstracted_units,
            "raw_to_canonical": raw_to_canonical,
            "canonical_to_abstracted": abstraction_step["canonical_to_abstracted"],
            "unit_abstractions": abstraction_step["unit_abstractions"],
            "pair_candidates": abstraction_step["pair_candidates"],
            "surviving_unit_ids": [unit["unit_id"] for unit in abstracted_units if unit["gate_status"] != "rejected_by_toxicity"],
            "rejected_unit_ids": [unit["unit_id"] for unit in abstracted_units if unit["gate_status"] == "rejected_by_toxicity"],
        }

    def _build_unit_docs(self, documents: List[Dict], unit: Dict) -> List[Dict]:
        unit_docs = []
        for doc_idx in unit.get("active_doc_indices", []):
            if doc_idx < 0 or doc_idx >= len(documents):
                continue
            document = documents[doc_idx]
            unit_docs.append(
                {
                    "doc_index": doc_idx + 1,
                    "doc_id": document.get("id"),
                    "claim": unit.get("claim", "NONE"),
                    "text": _truncate_text(document.get("context", ""), max_chars=350),
                }
            )
        return unit_docs


class ContradictionTypeClassifier:
    def __init__(self, llm_model):
        self.llm_model = llm_model

    def _default_pair_entry(self, pair: Dict) -> Dict:
        return {
            "pair_id": pair["pair_id"],
            "unit1": pair["unit1"],
            "unit2": pair["unit2"],
            "unit1_claim": pair.get("unit1_claim", "NONE"),
            "unit2_claim": pair.get("unit2_claim", "NONE"),
            "cot_explanation": pair.get("cot_explanation", ""),
            "contradiction_type": "uncertain",
            "rationale": "Fallback to uncertain because classification parsing failed.",
        }

    def _parse_response(self, response: str, pairs: List[Dict]) -> Dict[int, Dict]:
        cleaned = _strip_code_fences(response)
        payload = None
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = _extract_top_level_json(cleaned)

        results = {}
        if not payload:
            return results

        raw_pairs = payload.get("pairs", [])
        if not isinstance(raw_pairs, list):
            return results

        pair_lookup = {pair["pair_id"]: pair for pair in pairs}
        for item in raw_pairs:
            try:
                pair_id = int(item.get("pair_id"))
            except Exception:
                continue
            if pair_id not in pair_lookup:
                continue
            pair = pair_lookup[pair_id]
            results[pair_id] = {
                **self._default_pair_entry(pair),
                "contradiction_type": _normalize_type_label(item.get("contradiction_type")),
                "rationale": str(item.get("rationale", "")).strip() or "No rationale provided.",
            }
        return results

    def _classify_once(self, query: str, pairs: List[Dict]) -> Dict[int, Dict]:
        prompt = format_contradiction_type_classification_prompt(query, pairs)
        response = self.llm_model.query(prompt)
        return self._parse_response(response, pairs)

    def classify_pairs(self, query: str, pairs: List[Dict]) -> Dict:
        if not pairs:
            return {
                "step": "contradiction_type_classification",
                "typed_pairs": [],
                "factual_pairs": [],
            }

        results = {}
        try:
            results.update(self._classify_once(query, pairs))
        except Exception:
            results = {}

        missing_pairs = [pair for pair in pairs if pair["pair_id"] not in results]
        for pair in missing_pairs:
            try:
                results.update(self._classify_once(query, [pair]))
            except Exception:
                results[pair["pair_id"]] = self._default_pair_entry(pair)

        typed_pairs = [results[pair["pair_id"]] for pair in pairs]
        factual_pairs = [pair for pair in typed_pairs if pair["contradiction_type"] == "factual"]
        return {
            "step": "contradiction_type_classification",
            "typed_pairs": typed_pairs,
            "factual_pairs": factual_pairs,
        }


class InternalKnowledgeNoteGenerator:
    def __init__(self, llm_model, zero_temp_query_fn=None):
        self.llm_model = llm_model
        self.zero_temp_query_fn = zero_temp_query_fn

    def _default_note(self) -> Dict:
        return {
            "tentative_answer": "I don't know",
            "key_facts": [],
            "confidence": "low",
        }

    def generate(self, query: str) -> Dict:
        prompt = format_internal_knowledge_note_prompt(query)
        payload = None
        try:
            query_fn = self.zero_temp_query_fn or self.llm_model.query
            response = query_fn(prompt)
            cleaned = _strip_code_fences(response)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    payload = parsed
            except json.JSONDecodeError:
                payload = _extract_top_level_json(cleaned)
        except Exception:
            payload = None

        note = self._default_note()
        if payload:
            key_facts = payload.get("key_facts", [])
            if not isinstance(key_facts, list):
                key_facts = []
            note = {
                "tentative_answer": str(payload.get("tentative_answer", "I don't know")).strip() or "I don't know",
                "key_facts": [str(item).strip() for item in key_facts if str(item).strip()],
                "confidence": _normalize_confidence(payload.get("confidence")),
            }

        return {
            "step": "internal_knowledge_note",
            "note": note,
        }


class EvidenceAwareJudge:
    def __init__(self, llm_model):
        self.llm_model = llm_model

    def _default_unit_entry(self, unit: Dict, query_form: str, rationale: str = "") -> Dict:
        return {
            "unit_id": unit["unit_id"],
            "claim": unit.get("claim", "NONE"),
            "doc_indices": unit.get("doc_indices", []),
            "query_form": query_form,
            "IK_u": 0.5,
            "EV_raw": 0.5,
            "EV_u": 0.5,
            "query_restatement_risk": 0.5,
            "linkage_gap_risk": 0.5,
            "group_redundancy_risk": self._estimate_group_redundancy_risk(unit),
            "support_pattern": "unclear",
            "support_nature": "unclear",
            "rationale": rationale or "Fallback scores because judging parsing failed.",
        }

    def _parse_response(self, response: str, units: List[Dict], query_form: str) -> Dict[int, Dict]:
        cleaned = _strip_code_fences(response)
        payload = None
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = _extract_top_level_json(cleaned)

        results = {}
        if not payload:
            return results

        raw_units = payload.get("units", [])
        if not isinstance(raw_units, list):
            return results

        unit_lookup = {item["unit_id"]: item for item in units}
        for item in raw_units:
            try:
                unit_id = int(item.get("unit_id"))
            except Exception:
                continue
            if unit_id not in unit_lookup:
                continue
            unit = unit_lookup[unit_id]
            support_pattern = _normalize_support_pattern(
                item.get("support_pattern", item.get("support_nature"))
            )
            results[unit_id] = {
                **self._default_unit_entry(unit, query_form=query_form),
                "IK_u": _clamp_score(item.get("IK_u"), default=0.5),
                "EV_raw": _clamp_score(item.get("EV_raw", item.get("EV_u")), default=0.5),
                "EV_u": _clamp_score(item.get("EV_raw", item.get("EV_u")), default=0.5),
                "query_restatement_risk": _clamp_score(item.get("query_restatement_risk"), default=0.5),
                "linkage_gap_risk": _clamp_score(item.get("linkage_gap_risk"), default=0.5),
                "group_redundancy_risk": self._estimate_group_redundancy_risk(unit),
                "support_pattern": support_pattern,
                "support_nature": _support_pattern_to_nature(support_pattern),
                "rationale": str(item.get("rationale", "")).strip() or "No rationale provided.",
            }
        return results

    def _judge_once(self, query: str, query_form: str, internal_note: Dict, units: List[Dict]) -> Dict[int, Dict]:
        prompt = format_evidence_aware_judging_prompt(query, query_form, internal_note, units)
        response = self.llm_model.query(prompt)
        return self._parse_response(response, units, query_form)

    def _estimate_group_redundancy_risk(self, unit: Dict) -> float:
        docs = unit.get("docs", []) or []
        if len(docs) <= 1:
            return 0.0

        normalized_texts = []
        token_sets = []
        for doc in docs:
            text = _normalize_claim_key(doc.get("document_text", ""))
            if not text:
                continue
            normalized_texts.append(text)
            token_sets.append(set(token for token in text.split() if len(token) > 2))

        if len(normalized_texts) <= 1:
            return 0.0

        unique_ratio = len(set(normalized_texts)) / max(1, len(normalized_texts))
        exact_duplicate_ratio = 1.0 - unique_ratio

        pairwise_overlaps = []
        for left_idx in range(len(token_sets)):
            for right_idx in range(left_idx + 1, len(token_sets)):
                union = token_sets[left_idx] | token_sets[right_idx]
                if not union:
                    continue
                overlap = len(token_sets[left_idx] & token_sets[right_idx]) / len(union)
                pairwise_overlaps.append(overlap)

        avg_overlap = sum(pairwise_overlaps) / len(pairwise_overlaps) if pairwise_overlaps else 0.0
        max_overlap = max(pairwise_overlaps) if pairwise_overlaps else 0.0
        redundancy_risk = (
            0.40 * exact_duplicate_ratio
            + 0.35 * avg_overlap
            + 0.25 * max_overlap
        )
        if len(normalized_texts) >= 4 and avg_overlap >= 0.18:
            redundancy_risk += 0.10
        return max(0.0, min(1.0, redundancy_risk))

    def judge_units(self, query: str, internal_note: Dict, units: List[Dict]) -> Dict:
        query_form = infer_query_form(query)
        if not units:
            return {
                "step": "evidence_aware_judging",
                "query_form": query_form,
                "units": [],
            }

        results = {}
        try:
            results.update(self._judge_once(query, query_form, internal_note, units))
        except Exception:
            results = {}

        missing_units = [unit for unit in units if unit["unit_id"] not in results]
        for unit in missing_units:
            try:
                single_result = self._judge_once(query, query_form, internal_note, [unit])
                if unit["unit_id"] in single_result:
                    results.update(single_result)
                else:
                    results[unit["unit_id"]] = self._default_unit_entry(
                        unit,
                        query_form=query_form,
                        rationale="Fallback scores because single-unit evidence judging returned no matching unit.",
                    )
            except Exception:
                results[unit["unit_id"]] = self._default_unit_entry(unit, query_form=query_form)

        for unit in units:
            results.setdefault(
                unit["unit_id"],
                self._default_unit_entry(
                    unit,
                    query_form=query_form,
                    rationale="Fallback scores because evidence judging did not return this unit.",
                ),
            )

        judged = []
        for unit in units:
            unit_id = unit["unit_id"]
            entry = results.get(unit_id)
            if entry is None:
                entry = self._default_unit_entry(
                    unit,
                    query_form=query_form,
                    rationale="Fallback scores because evidence judging result lookup missed this unit at final assembly.",
                )
                results[unit_id] = entry
            judged.append(entry)
        return {
            "step": "evidence_aware_judging",
            "query_form": query_form,
            "units": judged,
        }


class KnowledgeAwareConflictResolver:
    def __init__(
        self,
        ik_weight: float = 0.5,
        ev_weight: float = 0.5,
        margin: float = 0.1,
        disable_credibility_scoring: bool = False,
        disable_internal_knowledge: bool = False,
        disable_evidence_scoring: bool = False,
    ):
        total = ik_weight + ev_weight
        if total <= 0:
            ik_weight = ev_weight = 0.5
            total = 1.0
        self.ik_weight = ik_weight / total
        self.ev_weight = ev_weight / total
        self.margin = float(margin)
        self.disable_credibility_scoring = bool(disable_credibility_scoring)
        self.disable_internal_knowledge = bool(disable_internal_knowledge) or self.disable_credibility_scoring
        self.disable_evidence_scoring = bool(disable_evidence_scoring) or self.disable_credibility_scoring

    def _effective_weights(self, internal_note: Dict) -> Dict:
        confidence = _normalize_confidence((internal_note or {}).get("confidence"))
        if self.disable_internal_knowledge and self.disable_evidence_scoring:
            return {
                "ik_weight": 0.0,
                "ev_weight": 0.0,
                "ik_used": False,
                "internal_note_confidence": "ablation_disabled",
            }
        if self.disable_internal_knowledge:
            return {
                "ik_weight": 0.0,
                "ev_weight": 1.0,
                "ik_used": False,
                "internal_note_confidence": "ablation_disabled",
            }
        if self.disable_evidence_scoring:
            if confidence == "low":
                return {
                    "ik_weight": 0.0,
                    "ev_weight": 0.0,
                    "ik_used": False,
                    "internal_note_confidence": confidence,
                }
            return {
                "ik_weight": 1.0,
                "ev_weight": 0.0,
                "ik_used": True,
                "internal_note_confidence": confidence,
            }
        if confidence == "low":
            return {
                "ik_weight": 0.0,
                "ev_weight": 1.0,
                "ik_used": False,
                "internal_note_confidence": confidence,
            }
        total = self.ik_weight + self.ev_weight
        return {
            "ik_weight": self.ik_weight / total,
            "ev_weight": self.ev_weight / total,
            "ik_used": True,
            "internal_note_confidence": confidence,
        }

    def _build_unit_scores(
        self,
        query_form: str,
        canonical_units: List[Dict],
        judged_units: List[Dict],
        internal_note: Dict,
    ) -> Dict[int, Dict]:
        if self.disable_credibility_scoring:
            return self._build_heuristic_unit_scores(query_form, canonical_units)

        judged_by_id = {item["unit_id"]: item for item in judged_units}
        weights = self._effective_weights(internal_note)
        unit_scores = {}

        for unit in canonical_units:
            unit_id = unit["unit_id"]
            judged = judged_by_id.get(unit_id)
            if unit.get("gate_status") == "rejected_by_toxicity":
                unit_scores[unit_id] = {
                    "unit_id": unit_id,
                    "claim": unit.get("claim", "NONE"),
                    "raw_unit_ids": unit.get("raw_unit_ids", []),
                    "doc_indices": [idx + 1 for idx in unit.get("active_doc_indices", [])],
                    "query_form": query_form,
                    "toxicity_score_max": unit.get("toxicity_score_max", 0.5),
                    "IK_u": 0.0,
                    "EV_u": 0.0,
                    "EV_u_raw": 0.0,
                    "query_restatement_risk": 0.0,
                    "linkage_gap_risk": 0.0,
                    "group_redundancy_risk": 0.0,
                    "support_pattern": "unclear",
                    "support_nature": "unclear",
                    "R_u": 0.0,
                    "ik_used": weights["ik_used"],
                    "status": "toxicity_rejected",
                    "rationale": "Unit was removed by the toxicity coarse filter.",
                }
                continue

            if judged is None:
                ik_score = 0.5
                ev_score_raw = 0.5
                query_restatement_risk = 0.5
                linkage_gap_risk = 0.5
                group_redundancy_risk = 0.0
                support_pattern = "unclear"
                support_nature = "unclear"
                rationale = "Fallback scores because unit judging was unavailable."
            else:
                ik_score = _clamp_score(judged.get("IK_u"), default=0.5)
                ev_score_raw = _clamp_score(judged.get("EV_raw", judged.get("EV_u")), default=0.5)
                query_restatement_risk = _clamp_score(judged.get("query_restatement_risk"), default=0.5)
                linkage_gap_risk = _clamp_score(judged.get("linkage_gap_risk"), default=0.5)
                group_redundancy_risk = _clamp_score(judged.get("group_redundancy_risk"), default=0.0)
                support_pattern = _normalize_support_pattern(judged.get("support_pattern", judged.get("support_nature")))
                support_nature = _normalize_support_nature(judged.get("support_nature"))
                rationale = judged.get("rationale", "")

            if self.disable_internal_knowledge:
                ik_score = 0.0
            if self.disable_evidence_scoring:
                ev_score_raw = 0.0
                ev_score_effective = 0.0
            else:
                ev_score_effective = self._adjust_evidence_score(
                    query_form,
                    ev_score_raw,
                    query_restatement_risk,
                    linkage_gap_risk,
                    group_redundancy_risk,
                    support_pattern,
                    support_nature,
                )
            reliability = weights["ik_weight"] * ik_score + weights["ev_weight"] * ev_score_effective
            unit_scores[unit_id] = {
                "unit_id": unit_id,
                "claim": unit.get("claim", "NONE"),
                "raw_unit_ids": unit.get("raw_unit_ids", []),
                "doc_indices": [idx + 1 for idx in unit.get("active_doc_indices", [])],
                "query_form": query_form,
                "toxicity_score_max": unit.get("toxicity_score_max", 0.5),
                "IK_u": ik_score,
                "EV_u": ev_score_effective,
                "EV_u_raw": ev_score_raw,
                "query_restatement_risk": query_restatement_risk,
                "linkage_gap_risk": linkage_gap_risk,
                "group_redundancy_risk": group_redundancy_risk,
                "support_pattern": support_pattern,
                "support_nature": support_nature,
                "R_u": reliability,
                "ik_used": weights["ik_used"],
                "status": "candidate",
                "rationale": rationale,
            }

        return unit_scores

    def _build_heuristic_unit_scores(self, query_form: str, canonical_units: List[Dict]) -> Dict[int, Dict]:
        max_doc_rank = 1
        for unit in canonical_units:
            doc_indices = unit.get("active_doc_indices") or unit.get("all_doc_indices") or []
            if doc_indices:
                max_doc_rank = max(max_doc_rank, max(doc_indices) + 1)

        unit_scores = {}
        for unit in canonical_units:
            unit_id = unit["unit_id"]
            active_doc_indices = unit.get("active_doc_indices", [])
            all_doc_indices = unit.get("all_doc_indices", active_doc_indices)
            if unit.get("gate_status") == "rejected_by_toxicity":
                baseline_score = 0.0
                status = "toxicity_rejected"
                rationale = "Unit was removed by the toxicity coarse filter."
            elif not active_doc_indices:
                baseline_score = 0.0
                status = "candidate"
                rationale = "No active evidence documents were available for the heuristic scoring baseline."
            else:
                support_count = len(active_doc_indices)
                support_component = min(1.0, math.log1p(support_count) / math.log1p(5))
                best_rank = min(active_doc_indices) + 1
                rank_component = 1.0 - ((best_rank - 1) / max(1, max_doc_rank))
                baseline_score = max(0.0, min(1.0, 0.25 + 0.45 * rank_component + 0.30 * support_component))
                status = "candidate"
                rationale = (
                    "Ablation baseline score from retrieval rank and number of active supporting documents; "
                    "IK note and evidence-aware judging were disabled."
                )

            unit_scores[unit_id] = {
                "unit_id": unit_id,
                "claim": unit.get("claim", "NONE"),
                "raw_unit_ids": unit.get("raw_unit_ids", []),
                "doc_indices": [idx + 1 for idx in active_doc_indices],
                "query_form": query_form,
                "toxicity_score_max": unit.get("toxicity_score_max", 0.5),
                "IK_u": 0.0,
                "EV_u": baseline_score,
                "EV_u_raw": baseline_score,
                "query_restatement_risk": 0.0,
                "linkage_gap_risk": 0.0,
                "group_redundancy_risk": 0.0,
                "support_pattern": "heuristic_rank_support",
                "support_nature": "heuristic_rank_support",
                "R_u": baseline_score,
                "ik_used": False,
                "status": status,
                "rationale": rationale,
                "all_doc_indices": [idx + 1 for idx in all_doc_indices],
            }

        return unit_scores

    def _adjust_evidence_score(
        self,
        query_form: str,
        ev_score_raw: float,
        query_restatement_risk: float,
        linkage_gap_risk: float,
        group_redundancy_risk: float,
        support_pattern: str,
        support_nature: str,
    ) -> float:
        ev_score = _clamp_score(ev_score_raw, default=0.5)
        ev_score -= 0.14 * query_restatement_risk
        ev_score -= 0.18 * linkage_gap_risk
        ev_score -= 0.16 * group_redundancy_risk

        high_restatement = query_restatement_risk >= 0.75
        high_linkage_gap = linkage_gap_risk >= 0.70
        strong_group_redundancy = group_redundancy_risk >= 0.55

        if support_pattern == "self_consistent_narrative":
            cap = 0.25
            if high_restatement or high_linkage_gap:
                cap = 0.20
            if strong_group_redundancy:
                cap = min(cap, 0.18)
            ev_score = min(ev_score, cap)
        elif support_pattern == "generic_background":
            cap = 0.45 if query_form in {"relation_lookup", "explanatory"} else 0.22
            ev_score = min(ev_score, cap)
        elif support_pattern in {"direct_slot_statement", "supported_relation_chain"}:
            if query_form in {"direct_slot", "identity_definition"}:
                protected_floor = 0.68 if support_pattern == "direct_slot_statement" else 0.58
                if high_restatement and (high_linkage_gap or strong_group_redundancy):
                    ev_score = min(ev_score, 0.40)
                elif high_restatement or high_linkage_gap:
                    ev_score = max(ev_score, min(0.52, protected_floor - 0.16))
                else:
                    ev_score = max(ev_score, protected_floor)
            else:
                ev_score = max(ev_score, max(0.0, ev_score_raw - 0.08 * linkage_gap_risk))
        elif support_nature == "mixed":
            if high_restatement or high_linkage_gap:
                ev_score = min(ev_score, 0.45)
            else:
                ev_score = min(ev_score, 0.72)
        else:
            if high_restatement and high_linkage_gap:
                ev_score = min(ev_score, 0.30)
            else:
                ev_score = min(ev_score, 0.60)

        return max(0.0, min(1.0, ev_score))

    def _build_doc_scores(
        self,
        documents: List[Dict],
        canonical_units: List[Dict],
        unit_scores: Dict[int, Dict],
    ) -> Dict[int, Dict]:
        doc_scores = {}
        doc_to_unit = {}
        for unit in canonical_units:
            for doc_idx in unit.get("all_doc_indices", []):
                doc_to_unit[doc_idx + 1] = unit["unit_id"]

        for idx, document in enumerate(documents, start=1):
            unit_id = doc_to_unit.get(idx)
            if unit_id is None:
                doc_scores[idx] = {
                    "doc_index": idx,
                    "doc_id": document.get("id"),
                    "claim": "NONE",
                    "IK_i": 0.0,
                    "EV_i": 0.0,
                    "R_i": 0.0,
                    "status": "unmapped",
                    "rationale": "Document was not mapped into any canonical support unit.",
                    "unit_id": None,
                }
                continue
            unit_score = unit_scores[unit_id]
            doc_scores[idx] = {
                "doc_index": idx,
                "doc_id": document.get("id"),
                "claim": unit_score.get("claim", "NONE"),
                "IK_i": unit_score.get("IK_u", 0.0),
                "EV_i": unit_score.get("EV_u", 0.0),
                "R_i": unit_score.get("R_u", 0.0),
                "status": unit_score.get("status", "candidate"),
                "rationale": unit_score.get("rationale", ""),
                "unit_id": unit_id,
            }
        return doc_scores

    def resolve(
        self,
        query: str,
        documents: List[Dict],
        canonical_units: List[Dict],
        judged_units: List[Dict],
        factual_pairs: List[Dict],
        internal_note: Dict,
    ) -> Dict:
        query_form = infer_query_form(query)
        if judged_units:
            query_form = judged_units[0].get("query_form", query_form)
        unit_scores = self._build_unit_scores(query_form, canonical_units, judged_units, internal_note)
        doc_scores = self._build_doc_scores(documents, canonical_units, unit_scores)
        candidate_unit_ids = [
            unit["unit_id"]
            for unit in canonical_units
            if unit.get("gate_status") != "rejected_by_toxicity"
        ]
        rejected_unit_ids = {
            unit["unit_id"]
            for unit in canonical_units
            if unit.get("gate_status") == "rejected_by_toxicity"
        }

        graph = nx.Graph()
        for unit_id in candidate_unit_ids:
            graph.add_node(unit_id)

        factual_edges = []
        candidate_set = set(candidate_unit_ids)
        for pair in factual_pairs:
            unit1 = int(pair["unit1"])
            unit2 = int(pair["unit2"])
            if unit1 not in candidate_set or unit2 not in candidate_set or unit1 == unit2:
                continue
            if not graph.has_edge(unit1, unit2):
                graph.add_edge(unit1, unit2, explanation=pair.get("rationale") or pair.get("cot_explanation", ""))
            factual_edges.append(
                {
                    "unit1": unit1,
                    "unit2": unit2,
                    "unit1_claim": pair.get("unit1_claim", ""),
                    "unit2_claim": pair.get("unit2_claim", ""),
                    "explanation": pair.get("rationale") or pair.get("cot_explanation", ""),
                }
            )

        retained = []
        filtered = set(rejected_unit_ids)
        unresolved_conflicts = []
        unresolved_pairs = set()

        ordered_candidates = sorted(
            candidate_unit_ids,
            key=lambda unit_id: (-unit_scores[unit_id]["R_u"], unit_id),
        )

        if graph.number_of_edges() == 0:
            singleton_candidate_mode = len(candidate_unit_ids) == 1
            for unit_id in ordered_candidates:
                score_item = unit_scores[unit_id]
                ev_is_zero = score_item.get("EV_u", 0.0) <= 1e-8
                high_restatement = score_item.get("query_restatement_risk", 0.0) >= 0.85
                if singleton_candidate_mode and ev_is_zero and high_restatement:
                    filtered.add(unit_id)
                    continue
                retained.append(unit_id)
        else:
            for unit_id in ordered_candidates:
                conflicting_retained = [other for other in retained if graph.has_edge(unit_id, other)]
                if not conflicting_retained:
                    retained.append(unit_id)
                    continue

                max_gap = max(unit_scores[other]["R_u"] - unit_scores[unit_id]["R_u"] for other in conflicting_retained)
                if max_gap > self.margin:
                    filtered.add(unit_id)
                    continue

                retained.append(unit_id)
                for other in conflicting_retained:
                    key = tuple(sorted((unit_id, other)))
                    if key in unresolved_pairs:
                        continue
                    unresolved_pairs.add(key)
                    edge_data = graph.get_edge_data(unit_id, other, default={})
                    unresolved_conflicts.append(
                        {
                            "unit1": key[0],
                            "unit2": key[1],
                            "unit1_claim": unit_scores[key[0]].get("claim", ""),
                            "unit2_claim": unit_scores[key[1]].get("claim", ""),
                            "score_gap": abs(unit_scores[other]["R_u"] - unit_scores[unit_id]["R_u"]),
                            "explanation": edge_data.get("explanation", "Retained both because the score gap was small."),
                        }
                    )

        retained_set = set(retained)
        filtered.update(unit_id for unit_id in candidate_unit_ids if unit_id not in retained_set)
        unresolved_member_set = {item["unit1"] for item in unresolved_conflicts} | {item["unit2"] for item in unresolved_conflicts}

        for unit_id, score_item in unit_scores.items():
            if unit_id in rejected_unit_ids:
                score_item["status"] = "toxicity_rejected"
            elif unit_id in retained_set and unit_id in unresolved_member_set:
                score_item["status"] = "retained_unresolved"
            elif unit_id in retained_set and graph.number_of_edges() == 0:
                score_item["status"] = "retained_no_factual_conflict"
            elif unit_id in filtered and graph.number_of_edges() == 0:
                score_item["status"] = "filtered_no_factual_conflict"
            elif unit_id in retained_set:
                score_item["status"] = "retained"
            elif unit_id in filtered:
                score_item["status"] = "filtered_conflict"

        final_documents = []
        retained_doc_ids = []
        filtered_doc_ids = []
        retained_doc_key_set = set()

        for unit in sorted(canonical_units, key=lambda item: item["unit_id"]):
            if unit["unit_id"] not in retained_set:
                continue
            for doc_idx in unit.get("active_doc_indices", []):
                if doc_idx < 0 or doc_idx >= len(documents):
                    continue
                document = documents[doc_idx]
                doc_key = document.get("id") or (document.get("context", "")[:128], doc_idx)
                if doc_key in retained_doc_key_set:
                    continue
                retained_doc_key_set.add(doc_key)
                final_documents.append(document)
                if document.get("id") is not None:
                    retained_doc_ids.append(document.get("id"))

        for unit in sorted(canonical_units, key=lambda item: item["unit_id"]):
            if unit["unit_id"] not in filtered:
                continue
            for doc_idx in unit.get("all_doc_indices", []):
                if doc_idx < 0 or doc_idx >= len(documents):
                    continue
                document = documents[doc_idx]
                doc_id = document.get("id")
                if doc_id is not None and doc_id not in filtered_doc_ids and doc_id not in retained_doc_ids:
                    filtered_doc_ids.append(doc_id)

        for doc_index, score_item in doc_scores.items():
            unit_id = score_item.get("unit_id")
            if unit_id is None:
                continue
            score_item["status"] = unit_scores[unit_id]["status"]

        if not final_documents:
            decision = "no_retained_documents"
        elif graph.number_of_edges() == 0:
            decision = "no_factual_conflict"
        elif unresolved_conflicts:
            decision = "unresolved_factual_conflict"
        elif len(retained_set) == 1:
            decision = "resolved_single_viewpoint"
        else:
            decision = "resolved_multi_viewpoint"

        weights = self._effective_weights(internal_note)
        return {
            "step": "knowledge_aware_conflict_resolution",
            "weights": {
                "IK_u": weights["ik_weight"],
                "EV_u": weights["ev_weight"],
                "ik_used": weights["ik_used"],
                "internal_note_confidence": weights["internal_note_confidence"],
            },
            "query_form": query_form,
            "resolution_margin": self.margin,
            "factual_edges": factual_edges,
            "retained_unit_ids": sorted(retained_set),
            "filtered_unit_ids": sorted(filtered),
            "retained_doc_ids": retained_doc_ids,
            "filtered_doc_ids": filtered_doc_ids,
            "unresolved_conflicts": sorted(unresolved_conflicts, key=lambda item: (item["unit1"], item["unit2"])),
            "unit_scores": [unit_scores[unit["unit_id"]] for unit in sorted(canonical_units, key=lambda item: item["unit_id"])],
            "doc_scores": [doc_scores[idx] for idx in sorted(doc_scores)],
            "final_documents": final_documents,
            "decision": decision,
        }
