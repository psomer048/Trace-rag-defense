"""
Defense Framework for PoisonedRAG.

This version keeps the existing hierarchical contradiction detector intact,
uses toxicity as a coarse first-stage filter, consolidates duplicate support
units, and then runs a unit-level factual conflict pipeline:
- contradiction type classification
- internal knowledge note generation
- per-unit evidence-aware judging
- knowledge-aware conflict resolution
"""

import json
import os
import re
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
from src.scorer_paths import ACTIVE_POISON_SCORER_REL_PATH
import numpy as np

from .poisoning_detection import PoisoningDetector
from .poison_traceback_expansion import PoisonTracebackExpander
from .factual_conflict_resolution import (
    ContradictionTypeClassifier,
    EvidenceAwareJudge,
    InternalKnowledgeNoteGenerator,
    KnowledgeAwareConflictResolver,
    SupportUnitConsolidator,
    infer_query_form,
)
from .prompts_defense import (
    format_cot_contradiction_prompt,
    format_per_document_candidate_extraction_prompt,
    format_viewpoint_joint_scoring_prompt,
)
from .visualization import ContradictionGraphVisualizer


VALID_CONTRIBUTION_TYPES = {
    "final_answer",
    "intermediate_step",
    "background",
    "attack_instruction",
    "none",
}
VALID_CONTRADICTION_TYPES = {"factual", "non-factual", "uncertain"}
VALID_RISK_FLAGS = {
    "query_copying",
    "single_doc_only",
    "no_corroboration",
    "synthetic_direct_answer",
    "conflicts_with_other_docs",
    "internally_implausible",
}
INVALID_ANSWER_MARKERS = {"", "NONE", "UNKNOWN", "IDK", "N/A", "NULL"}
CONTRADICTION_PRIORITY = {"uncertain": 0, "non-factual": 1, "factual": 2}
MODEL_ERROR_PREFIXES = (
    "GLM API Error:",
    "DeepSeek API Error:",
    "OpenRouter API Error:",
    "GPT_CA API Error:",
)


def _is_model_error_response(response) -> bool:
    text = str(response or "").strip()
    return any(text.startswith(prefix) for prefix in MODEL_ERROR_PREFIXES)


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _word_limit(text: str, max_words: int = 20) -> str:
    if not isinstance(text, str):
        return ""
    words = text.strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


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


def _extract_json_field(text: str, field_name: str):
    pattern = re.compile(rf'"{re.escape(field_name)}"\s*:', re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    idx = match.end()
    while idx < len(text) and text[idx].isspace():
        idx += 1
    segment = _extract_balanced_segment(text, idx)
    if not segment:
        return None
    try:
        return json.loads(segment)
    except json.JSONDecodeError:
        return None


def _clamp_score(value, default: float = 0.5) -> float:
    try:
        score = float(value)
    except Exception:
        score = default
    return max(0.0, min(1.0, score))


def _canonical_doc_key(doc: Dict) -> Tuple[str, str]:
    doc_id = doc.get("id")
    if doc_id is not None:
        return ("id", str(doc_id))
    return ("context", doc.get("context", ""))


def _normalize_flag(flag: str) -> Optional[str]:
    value = str(flag or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "single_doc": "single_doc_only",
        "single_document_only": "single_doc_only",
        "query_copy": "query_copying",
        "query_restatement": "query_copying",
        "synthetic_answer": "synthetic_direct_answer",
        "no_support": "no_corroboration",
        "no_corroborated_support": "no_corroboration",
        "conflict_with_other_docs": "conflicts_with_other_docs",
        "implausible": "internally_implausible",
    }
    value = aliases.get(value, value)
    return value if value in VALID_RISK_FLAGS else None


def _normalize_text_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())).strip()


class ContradictionDetector:
    """Hierarchical contradiction detection and normalization."""

    def __init__(self, llm_model, dataset_name: str = None, config: Optional[Dict] = None):
        self.llm_model = llm_model
        self.dataset_name = dataset_name
        self.config = config or {}

    def _normalize_contribution_type(self, raw_value) -> str:
        value = str(raw_value or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "final": "final_answer",
            "answer": "final_answer",
            "direct_answer": "final_answer",
            "intermediate": "intermediate_step",
            "bridge": "intermediate_step",
            "step": "intermediate_step",
            "context": "background",
            "contextual": "background",
            "contextual_info": "background",
            "background_info": "background",
            "prompt_injection": "attack_instruction",
            "prompt_injection_attack": "attack_instruction",
            "instruction_attack": "attack_instruction",
            "manipulative_instruction": "attack_instruction",
            "adversarial_instruction": "attack_instruction",
            "irrelevant": "none",
            "unknown": "none",
        }
        value = aliases.get(value, value)
        return value if value in VALID_CONTRIBUTION_TYPES else "none"

    def _normalize_contradiction_type(self, raw_value) -> str:
        value = str(raw_value or "").strip().lower().replace("_", "-")
        aliases = {
            "nonfactual": "non-factual",
            "non factual": "non-factual",
            "non_factual": "non-factual",
            "unclear": "uncertain",
        }
        value = aliases.get(value, value)
        return value if value in VALID_CONTRADICTION_TYPES else "uncertain"

    def _is_invalid_answer(self, answer: str) -> bool:
        return (answer or "").strip().upper() in INVALID_ANSWER_MARKERS

    def _normalize_doc_contributions(self, raw_items: List[Dict], num_documents: int) -> List[Dict]:
        normalized = {
            idx: {
                "index": idx,
                "contribution_type": "none",
                "answer_or_claim": "NONE",
            }
            for idx in range(num_documents)
        }

        for item in raw_items or []:
            try:
                raw_index = int(item.get("index")) - 1
            except Exception:
                continue
            if not (0 <= raw_index < num_documents):
                continue

            contribution_type = self._normalize_contribution_type(item.get("contribution_type"))
            answer_or_claim = str(
                item.get("answer_or_claim", item.get("answer", item.get("claim", "NONE")))
            ).strip()
            if contribution_type == "none":
                answer_or_claim = "NONE"
            elif not answer_or_claim:
                answer_or_claim = "NONE"

            normalized[raw_index] = {
                "index": raw_index,
                "contribution_type": contribution_type,
                "answer_or_claim": answer_or_claim,
            }

        return [normalized[idx] for idx in range(num_documents)]

    def _build_singleton_support_units(self, doc_contributions: List[Dict]) -> Tuple[List[Dict], Dict[int, int]]:
        units = []
        raw_to_norm = {}
        for item in doc_contributions:
            contribution_type = item.get("contribution_type")
            answer = str(item.get("answer_or_claim", "")).strip()
            if contribution_type not in {"final_answer", "intermediate_step"}:
                continue
            if self._is_invalid_answer(answer):
                continue
            raw_unit_id = item["index"] + 1
            raw_to_norm[raw_unit_id] = len(units)
            units.append({
                "unit_id": len(units),
                "docs": [item["index"]],
                "derived_answer": answer,
            })
        return units, raw_to_norm

    def _backfill_missing_final_answer_units(
        self,
        normalized_units: List[Dict],
        raw_to_norm: Dict[int, int],
        doc_contributions: List[Dict],
    ) -> Tuple[List[Dict], Dict[int, int]]:
        if not bool(self.config.get("backfill_missing_final_answer_units", False)):
            return normalized_units, raw_to_norm

        covered_doc_indices = set()
        for unit in normalized_units:
            for doc_idx in unit.get("docs", []):
                covered_doc_indices.add(doc_idx)

        next_raw_unit_id = max(raw_to_norm.keys(), default=0)
        for item in doc_contributions:
            if item.get("contribution_type") != "final_answer":
                continue
            doc_idx = item.get("index")
            if doc_idx in covered_doc_indices:
                continue
            answer = str(item.get("answer_or_claim", "")).strip()
            if self._is_invalid_answer(answer):
                continue

            next_raw_unit_id += 1
            raw_to_norm[next_raw_unit_id] = len(normalized_units)
            normalized_units.append({
                "unit_id": len(normalized_units),
                "docs": [doc_idx],
                "derived_answer": answer,
            })
            covered_doc_indices.add(doc_idx)

        return normalized_units, raw_to_norm

    def _normalize_support_units(
        self,
        raw_units: List[Dict],
        doc_contributions: List[Dict],
    ) -> Tuple[List[Dict], Dict[int, int]]:
        contribution_map = {item["index"]: item["contribution_type"] for item in doc_contributions}
        normalized_units = []
        raw_to_norm = {}

        for raw_unit in raw_units or []:
            try:
                raw_unit_id = int(raw_unit.get("unit_id", raw_unit.get("id", raw_unit.get("index"))))
            except Exception:
                continue
            if raw_unit_id in raw_to_norm:
                continue

            raw_docs = raw_unit.get("docs", raw_unit.get("documents", raw_unit.get("doc_indices", [])))
            if not isinstance(raw_docs, list):
                continue

            docs = []
            seen_docs = set()
            contains_attack_instruction = False
            for raw_doc_idx in raw_docs:
                try:
                    doc_idx = int(raw_doc_idx) - 1
                except Exception:
                    continue
                if doc_idx < 0 or doc_idx >= len(doc_contributions) or doc_idx in seen_docs:
                    continue
                if contribution_map.get(doc_idx, "none") == "attack_instruction":
                    contains_attack_instruction = True
                    continue
                seen_docs.add(doc_idx)
                docs.append(doc_idx)

            if not docs:
                continue

            contribution_types = [contribution_map.get(doc_idx, "none") for doc_idx in docs]
            if not any(ct in {"final_answer", "intermediate_step"} for ct in contribution_types):
                continue

            if contains_attack_instruction:
                remaining_claims = [
                    str(doc_contributions[doc_idx].get("answer_or_claim", "")).strip()
                    for doc_idx in docs
                    if contribution_map.get(doc_idx, "none") in {"final_answer", "intermediate_step"}
                ]
                remaining_claims = [
                    claim
                    for claim in remaining_claims
                    if not self._is_invalid_answer(claim)
                ]
                if not remaining_claims:
                    continue
                derived_answer = remaining_claims[0]
            else:
                derived_answer = str(
                    raw_unit.get("derived_answer", raw_unit.get("answer", raw_unit.get("claim", "")))
                ).strip()
                if self._is_invalid_answer(derived_answer):
                    continue

            raw_to_norm[raw_unit_id] = len(normalized_units)
            normalized_units.append({
                "unit_id": len(normalized_units),
                "docs": docs,
                "derived_answer": derived_answer,
            })

        if not normalized_units:
            return self._build_singleton_support_units(doc_contributions)

        normalized_units, raw_to_norm = self._backfill_missing_final_answer_units(
            normalized_units,
            raw_to_norm,
            doc_contributions,
        )

        return normalized_units, raw_to_norm

    def _normalize_contradictory_units(
        self,
        raw_conflicts: List[Dict],
        raw_unit_mapping: Dict[int, int],
        normalized_units: List[Dict],
        overall_explanation: str,
    ) -> List[Dict]:
        deduped = {}
        for conflict in raw_conflicts or []:
            try:
                raw_unit1 = int(conflict.get("unit1", conflict.get("index1")))
                raw_unit2 = int(conflict.get("unit2", conflict.get("index2")))
            except Exception:
                continue
            if raw_unit1 not in raw_unit_mapping or raw_unit2 not in raw_unit_mapping:
                continue

            unit1 = raw_unit_mapping[raw_unit1]
            unit2 = raw_unit_mapping[raw_unit2]
            if unit1 == unit2:
                continue
            if unit1 > unit2:
                unit1, unit2 = unit2, unit1

            unit1_claim = _normalize_text_for_match(normalized_units[unit1].get("derived_answer", ""))
            unit2_claim = _normalize_text_for_match(normalized_units[unit2].get("derived_answer", ""))
            if unit1_claim and unit1_claim == unit2_claim:
                continue

            contradiction_type = self._normalize_contradiction_type(conflict.get("type"))
            explanation = str(conflict.get("explanation", conflict.get("reason", overall_explanation or ""))).strip()
            if not explanation:
                explanation = "Conflict detected between candidate support units."

            key = (unit1, unit2)
            priority = CONTRADICTION_PRIORITY[contradiction_type]
            current = deduped.get(key)
            if current is None or priority > current["priority"]:
                deduped[key] = {
                    "unit1": unit1,
                    "unit2": unit2,
                    "type": contradiction_type,
                    "explanation": explanation,
                    "priority": priority,
                }

        normalized = []
        for item in deduped.values():
            item.pop("priority", None)
            normalized.append(item)
        normalized.sort(key=lambda x: (x["unit1"], x["unit2"]))
        return normalized

    def validate_cot_result(self, cot_result: Dict, num_documents: int) -> Dict:
        """Validate and normalize hierarchical CoT output."""
        overall_explanation = _word_limit(cot_result.get("overall_explanation", ""), max_words=20)
        doc_contributions = self._normalize_doc_contributions(
            cot_result.get("doc_contributions", []),
            num_documents,
        )
        support_units, raw_unit_mapping = self._normalize_support_units(
            cot_result.get("support_units", []),
            doc_contributions,
        )
        contradictory_units = self._normalize_contradictory_units(
            cot_result.get("contradictory_units", []),
            raw_unit_mapping,
            support_units,
            overall_explanation,
        )

        return {
            "has_contradiction": bool(contradictory_units),
            "doc_contributions": doc_contributions,
            "support_units": support_units,
            "contradictory_units": contradictory_units,
            "overall_explanation": overall_explanation,
        }

    def _parse_per_document_candidate_response(self, response: str) -> Dict:
        payload = None
        cleaned = _strip_code_fences(response)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = _extract_top_level_json(cleaned)

        if not isinstance(payload, dict):
            payload = {}

        candidate_answer = str(payload.get("candidate_answer", "NONE")).strip() or "NONE"
        claim = str(payload.get("claim", "NONE")).strip() or "NONE"
        evidence_span = str(payload.get("evidence_span", "NONE")).strip() or "NONE"
        def coerce_bool(value) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            return str(value or "").strip().lower() in {"true", "1", "yes", "y"}

        is_answer_bearing = coerce_bool(payload.get("is_answer_bearing", False))
        is_attack_instruction = coerce_bool(payload.get("is_attack_instruction", False))

        if self._is_invalid_answer(candidate_answer):
            candidate_answer = "NONE"
        if self._is_invalid_answer(claim):
            claim = "NONE"
        if self._is_invalid_answer(evidence_span):
            evidence_span = "NONE"

        if candidate_answer == "NONE" and claim == "NONE":
            is_answer_bearing = False

        return {
            "candidate_answer": candidate_answer,
            "claim": claim,
            "evidence_span": evidence_span,
            "is_answer_bearing": is_answer_bearing,
            "is_attack_instruction": is_attack_instruction,
        }

    def _extract_single_document_candidate(self, query: str, document: Dict, doc_index: int) -> Dict:
        prompt = format_per_document_candidate_extraction_prompt(query, document, doc_index)
        response = ""
        max_attempts = max(1, int(self.config.get("cot_model_error_retries", 2)) + 1)

        for attempt in range(1, max_attempts + 1):
            response = self.llm_model.query(prompt)
            if _is_model_error_response(response):
                print(f"[Warn] per-document candidate extraction model error attempt {attempt}/{max_attempts}: {response[:240]}")
                if attempt < max_attempts:
                    time.sleep(5.0 * attempt)
                    continue
            break

        parsed = self._parse_per_document_candidate_response(response)
        parsed["raw_response"] = response
        return parsed

    def detect_contradictions_per_document_candidates(self, query: str, documents: List[Dict]) -> Dict:
        """Build pseudo support units from per-document candidate extraction, without global CoT conflict analysis."""
        doc_contributions = []
        support_units = []
        per_document_candidates = []

        for doc_idx, document in enumerate(documents, start=1):
            candidate = self._extract_single_document_candidate(query, document, doc_idx)
            per_document_candidates.append({
                "index": doc_idx,
                **candidate,
            })

            if candidate.get("is_attack_instruction"):
                contribution_type = "attack_instruction"
                answer_or_claim = (
                    candidate.get("candidate_answer")
                    if not self._is_invalid_answer(candidate.get("candidate_answer", ""))
                    else candidate.get("claim", "attack instruction")
                )
            elif candidate.get("is_answer_bearing") and not self._is_invalid_answer(candidate.get("candidate_answer", "")):
                contribution_type = "final_answer"
                answer_or_claim = candidate.get("candidate_answer", "NONE")
            elif candidate.get("is_answer_bearing") and not self._is_invalid_answer(candidate.get("claim", "")):
                contribution_type = "intermediate_step"
                answer_or_claim = candidate.get("claim", "NONE")
            elif not self._is_invalid_answer(candidate.get("claim", "")):
                contribution_type = "background"
                answer_or_claim = candidate.get("claim", "NONE")
            else:
                contribution_type = "none"
                answer_or_claim = "NONE"

            doc_contributions.append({
                "index": doc_idx,
                "contribution_type": contribution_type,
                "answer_or_claim": answer_or_claim,
            })

            if contribution_type not in {"final_answer", "intermediate_step"}:
                continue

            support_units.append({
                "unit_id": len(support_units) + 1,
                "docs": [doc_idx],
                "derived_answer": answer_or_claim,
            })

        raw_result = {
            "has_contradiction": False,
            "doc_contributions": doc_contributions,
            "support_units": support_units,
            "contradictory_units": [],
            "overall_explanation": "Per-document candidates extracted without global semantic conflict CoT.",
        }
        normalized = self.validate_cot_result(raw_result, len(documents))
        normalized["ablation_mode"] = "per_document_candidate_extraction"
        normalized["per_document_candidates"] = per_document_candidates
        return normalized

    def _partial_parse_response(self, cleaned_response: str) -> Optional[Dict]:
        partial = {}
        for field_name in [
            "has_contradiction",
            "doc_contributions",
            "support_units",
            "contradictory_units",
            "overall_explanation",
        ]:
            value = _extract_json_field(cleaned_response, field_name)
            if value is not None:
                partial[field_name] = value
        return partial or None

    def detect_contradictions_cot(self, query: str, documents: List[Dict]) -> Dict:
        """Run the hierarchical contradiction prompt and normalize the output."""
        cot_prompt = format_cot_contradiction_prompt(query, documents, dataset_name=self.dataset_name)
        response = ""
        max_attempts = max(1, int(self.config.get("cot_model_error_retries", 2)) + 1)

        for attempt in range(1, max_attempts + 1):
            response = self.llm_model.query(cot_prompt)
            if _is_model_error_response(response):
                print(f"[Warn] contradiction_detection model error attempt {attempt}/{max_attempts}: {response[:240]}")
                if attempt < max_attempts:
                    time.sleep(5.0 * attempt)
                    continue

            try:
                debug_dir = "results/debug_cot"
                os.makedirs(debug_dir, exist_ok=True)
                query_hash = str(hash(query))[:8]
                debug_file = os.path.join(debug_dir, f"cot_response_{query_hash}.txt")
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(f"Query: {query}\n\n")
                    f.write(f"Documents count: {len(documents)}\n\n")
                    f.write(f"CoT Prompt:\n{cot_prompt}\n\n")
                    f.write(f"LLM Response:\n{response}\n")

                cleaned_response = _strip_code_fences(response)

                json_result = None
                try:
                    parsed = json.loads(cleaned_response)
                    if isinstance(parsed, dict):
                        json_result = parsed
                except json.JSONDecodeError:
                    pass

                if json_result is None:
                    json_result = _extract_top_level_json(cleaned_response)

                if json_result is None:
                    json_result = self._partial_parse_response(cleaned_response)

                if json_result is None:
                    raise json.JSONDecodeError("Unable to parse hierarchical contradiction output", cleaned_response, 0)

                return self.validate_cot_result(json_result, len(documents))

            except Exception as exc:
                print(f"Error in contradiction detection: {exc}")
                traceback.print_exc()
                if attempt < max_attempts and _is_model_error_response(response):
                    time.sleep(5.0 * attempt)
                    continue

                debug_dir = "results/debug_cot"
                os.makedirs(debug_dir, exist_ok=True)
                query_hash = str(hash(query))[:8]
                debug_file = os.path.join(debug_dir, f"cot_parse_error_{query_hash}.txt")
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(f"Query: {query}\n\n")
                    f.write(f"Error: {exc}\n\n")
                    f.write(f"Raw Response:\n{response}\n")

                fallback = self._partial_parse_response(_strip_code_fences(response)) or {}
                return self.validate_cot_result(fallback, len(documents))

        fallback = self._partial_parse_response(_strip_code_fences(response)) or {}
        return self.validate_cot_result(fallback, len(documents))


class DefenseFramework:
    """Main defense framework."""

    def __init__(
        self,
        llm_model,
        config: Optional[Dict] = None,
        relevance_llm_model=None,
        embedding_model=None,
        tokenizer=None,
        get_emb_fn=None,
        debug_viewpoint_scores: bool = True,
        debug_dir: str = "logs/viewpoint_scores",
        dataset_name: str = None,
    ):
        self.llm_model = llm_model
        self.config = config or {}
        self.config.setdefault("toxicity_gate_threshold", 0.5)
        self.config.setdefault("toxicity_fallback_keep_k", 2)
        self.config.setdefault("support_weight", 0.5)
        self.config.setdefault("evidence_weight", 0.5)
        self.config.setdefault("ik_weight", 0.5)
        self.config.setdefault("ev_weight", 0.5)
        self.config.setdefault("resolution_margin", 0.1)
        self.config.setdefault("backfill_missing_final_answer_units", False)
        self.config.setdefault("enable_poison_traceback_expansion", False)
        self.config.setdefault("ablation_disable_coarse_filter", False)
        self.config.setdefault("ablation_disable_conflict_type_classification", False)
        self.config.setdefault("ablation_disable_credibility_scoring", False)
        self.config.setdefault("ablation_disable_internal_knowledge", False)
        self.config.setdefault("ablation_disable_evidence_scoring", False)
        self.config.setdefault("ablation_disable_global_semantic_conflict_cot", False)
        self.config.setdefault("ablation_claim_only_no_conflict_graph", False)
        self.config.setdefault("ablation_doc_level_filtering", False)
        self.config.setdefault("cot_model_error_retries", 2)
        self.debug_viewpoint_scores = debug_viewpoint_scores
        self.debug_dir = debug_dir
        self.dataset_name = dataset_name

        if self.debug_viewpoint_scores:
            os.makedirs(self.debug_dir, exist_ok=True)

        self.contradiction_detector = ContradictionDetector(
            llm_model,
            dataset_name=dataset_name,
            config=self.config,
        )
        self.poisoning_detector = PoisoningDetector(
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            get_emb_fn=get_emb_fn,
            debug=True,
            logger=print,
            model_path=self.config.get("ml_model_path", ACTIVE_POISON_SCORER_REL_PATH),
            collect_data_mode=bool(self.config.get("collect_data", False)),
            data_filepath=self.config.get("collect_data_path", "poison_training_data.csv"),
        )
        self.visualizer = ContradictionGraphVisualizer()
        self.enable_visualization = self.config.get("enable_visualization", False)
        self.visualization_dir = self.config.get("visualization_dir", "results/visualizations")
        self.support_unit_consolidator = SupportUnitConsolidator(llm_model)
        self.contradiction_type_classifier = ContradictionTypeClassifier(llm_model)
        self.internal_knowledge_note_generator = InternalKnowledgeNoteGenerator(
            llm_model,
            zero_temp_query_fn=self._query_with_temperature_zero,
        )
        self.evidence_aware_judge = EvidenceAwareJudge(llm_model)
        self.knowledge_aware_resolver = KnowledgeAwareConflictResolver(
            ik_weight=float(self.config.get("ik_weight", 0.5)),
            ev_weight=float(self.config.get("ev_weight", 0.5)),
            margin=float(self.config.get("resolution_margin", 0.1)),
            disable_credibility_scoring=bool(self.config.get("ablation_disable_credibility_scoring", False)),
            disable_internal_knowledge=bool(self.config.get("ablation_disable_internal_knowledge", False)),
            disable_evidence_scoring=bool(self.config.get("ablation_disable_evidence_scoring", False)),
        )
        self.poison_traceback_expander = PoisonTracebackExpander(self.config)

    def _query_with_temperature_zero(self, prompt: str) -> str:
        if not hasattr(self.llm_model, "temperature"):
            return self.llm_model.query(prompt)

        original_temperature = self.llm_model.temperature
        self.llm_model.temperature = 0.0
        try:
            return self.llm_model.query(prompt)
        finally:
            self.llm_model.temperature = original_temperature

    def arbitrate_viewpoints(self, query: str, viewpoints: List[Dict], retrieve_fn: Callable) -> Dict:
        return self.poisoning_detector.arbitrate_viewpoints(query, viewpoints, retrieve_fn)

    def _build_doc_claim_map(self, cot_result: Dict) -> Dict[int, Dict]:
        return {
            item["index"]: item
            for item in cot_result.get("doc_contributions", [])
        }

    def _split_attack_instruction_documents(
        self,
        documents: List[Dict],
        cot_result: Dict,
    ) -> Tuple[List[Dict], List[str], List[int]]:
        claim_map = self._build_doc_claim_map(cot_result)
        safe_documents = []
        filtered_doc_ids = []
        filtered_doc_indices = []
        for doc_idx, document in enumerate(documents):
            contribution = claim_map.get(doc_idx, {})
            if contribution.get("contribution_type") == "attack_instruction":
                filtered_doc_indices.append(doc_idx)
                if document.get("id") is not None:
                    filtered_doc_ids.append(document.get("id"))
                continue
            safe_documents.append(document)
        return safe_documents, filtered_doc_ids, filtered_doc_indices

    def _apply_attack_instruction_trace(
        self,
        result: Dict,
        documents: List[Dict],
        cot_result: Dict,
    ) -> None:
        claim_map = self._build_doc_claim_map(cot_result)
        attack_doc_indices = {
            idx
            for idx, item in claim_map.items()
            if item.get("contribution_type") == "attack_instruction"
        }
        if not attack_doc_indices:
            return

        attack_doc_ids = [
            documents[idx].get("id")
            for idx in sorted(attack_doc_indices)
            if 0 <= idx < len(documents) and documents[idx].get("id") is not None
        ]
        attack_doc_keys = {
            _canonical_doc_key(documents[idx])
            for idx in attack_doc_indices
            if 0 <= idx < len(documents)
        }

        final_documents = result.get("final_documents", [])
        if final_documents:
            result["final_documents"] = [
                document
                for document in final_documents
                if _canonical_doc_key(document) not in attack_doc_keys
            ]

        resolution_summary = result.setdefault("resolution_summary", {})
        retained_doc_ids = [
            document.get("id")
            for document in result.get("final_documents", [])
            if document.get("id") is not None
        ]
        resolution_summary["retained_doc_ids"] = retained_doc_ids
        filtered_doc_ids = resolution_summary.setdefault("filtered_doc_ids", [])
        retained_doc_id_set = set(retained_doc_ids)
        for doc_id in attack_doc_ids:
            if doc_id not in retained_doc_id_set and doc_id not in filtered_doc_ids:
                filtered_doc_ids.append(doc_id)

        for score_item in result.get("doc_reliability_scores", []):
            doc_idx = int(score_item.get("doc_index", 0)) - 1
            if doc_idx not in attack_doc_indices:
                continue
            contribution = claim_map.get(doc_idx, {})
            score_item["claim"] = contribution.get("answer_or_claim", "NONE")
            score_item["IK_i"] = 0.0
            score_item["EV_i"] = 0.0
            score_item["R_i"] = 0.0
            score_item["status"] = "filtered_attack_instruction"
            score_item["rationale"] = (
                "Filtered because the document was classified as assistant-directed attack instruction rather than evidence."
            )
            score_item["unit_id"] = None

        for claim_item in result.get("extracted_claims", []):
            doc_idx = int(claim_item.get("doc_index", 0)) - 1
            if doc_idx not in attack_doc_indices:
                continue
            contribution = claim_map.get(doc_idx, {})
            claim_item["claim"] = contribution.get("answer_or_claim", "NONE")
            claim_item["status"] = "filtered_attack_instruction"

        if not result.get("final_documents"):
            result["defense_decision"] = "no_retained_documents"

    def _build_doc_pair_candidates(self, documents: List[Dict], cot_result: Dict) -> List[Dict]:
        support_units = {
            unit["unit_id"]: unit
            for unit in cot_result.get("support_units", [])
        }
        claim_map = self._build_doc_claim_map(cot_result)
        pair_map = {}

        for conflict in cot_result.get("contradictory_units", []):
            unit1 = support_units.get(conflict.get("unit1"))
            unit2 = support_units.get(conflict.get("unit2"))
            if not unit1 or not unit2:
                continue

            for doc_idx1 in unit1.get("docs", []):
                for doc_idx2 in unit2.get("docs", []):
                    if doc_idx1 == doc_idx2:
                        continue
                    left, right = sorted((doc_idx1, doc_idx2))
                    key = (left, right)
                    pair_map.setdefault(key, []).append(str(conflict.get("explanation", "")).strip())

        pair_candidates = []
        for pair_id, (doc_idx1, doc_idx2) in enumerate(sorted(pair_map.keys()), start=1):
            explanations = [item for item in pair_map[(doc_idx1, doc_idx2)] if item]
            pair_candidates.append({
                "pair_id": pair_id,
                "doc1": doc_idx1 + 1,
                "doc2": doc_idx2 + 1,
                "doc1_index": doc_idx1,
                "doc2_index": doc_idx2,
                "doc1_id": documents[doc_idx1].get("id"),
                "doc2_id": documents[doc_idx2].get("id"),
                "doc1_claim": claim_map.get(doc_idx1, {}).get("answer_or_claim", "NONE"),
                "doc2_claim": claim_map.get(doc_idx2, {}).get("answer_or_claim", "NONE"),
                "doc1_text": documents[doc_idx1].get("context", ""),
                "doc2_text": documents[doc_idx2].get("context", ""),
                "cot_explanation": " | ".join(explanations),
            })
        return pair_candidates

    def _project_toxicity_to_docs(
        self,
        documents: List[Dict],
        viewpoints: List[Dict],
        toxicity_step: Dict,
        cot_result: Dict,
    ) -> Dict:
        doc_claim_map = self._build_doc_claim_map(cot_result)
        surviving_viewpoints = set(toxicity_step.get("surviving_viewpoints", []))
        rejected_viewpoints = set(toxicity_step.get("rejected_viewpoints", []))
        viewpoint_toxicity = {
            item["viewpoint_index"]: _clamp_score(item.get("toxicity_score"), default=0.5)
            for item in toxicity_step.get("viewpoint_toxicity", [])
        }

        doc_to_viewpoints = {}
        for viewpoint in viewpoints:
            for doc_idx in viewpoint.get("doc_indices", []):
                doc_to_viewpoints.setdefault(doc_idx, []).append(viewpoint["viewpoint_index"])

        doc_projection = []
        surviving_doc_indices = []
        rejected_doc_indices = []

        for doc_idx, document in enumerate(documents):
            linked_viewpoints = doc_to_viewpoints.get(doc_idx, [])
            contribution = doc_claim_map.get(doc_idx, {})
            if contribution.get("contribution_type") == "attack_instruction":
                gate_status = "rejected_attack_instruction"
                toxicity_score = 1.0
                rejected_doc_indices.append(doc_idx + 1)
            elif not linked_viewpoints:
                gate_status = "survive_unmapped"
                toxicity_score = 0.5
                surviving_doc_indices.append(doc_idx + 1)
            else:
                toxicity_score = max(
                    viewpoint_toxicity.get(viewpoint_idx, 0.5)
                    for viewpoint_idx in linked_viewpoints
                )
                if any(viewpoint_idx in surviving_viewpoints for viewpoint_idx in linked_viewpoints):
                    gate_status = "survive"
                    surviving_doc_indices.append(doc_idx + 1)
                elif all(viewpoint_idx in rejected_viewpoints for viewpoint_idx in linked_viewpoints):
                    gate_status = "rejected_by_toxicity"
                    rejected_doc_indices.append(doc_idx + 1)
                else:
                    gate_status = "survive"
                    surviving_doc_indices.append(doc_idx + 1)

            doc_projection.append({
                "doc_index": doc_idx + 1,
                "doc_id": document.get("id"),
                "claim": contribution.get("answer_or_claim", "NONE"),
                "contribution_type": contribution.get("contribution_type", "none"),
                "viewpoint_indices": linked_viewpoints,
                "toxicity_score": toxicity_score,
                "gate_status": gate_status,
            })

        return {
            "doc_projection": doc_projection,
            "surviving_doc_indices": surviving_doc_indices,
            "rejected_doc_indices": rejected_doc_indices,
        }

    def _build_judging_documents(
        self,
        documents: List[Dict],
        cot_result: Dict,
        surviving_doc_indices: List[int],
    ) -> List[Dict]:
        claim_map = self._build_doc_claim_map(cot_result)
        judging_documents = []
        for doc_index in surviving_doc_indices:
            zero_based_idx = doc_index - 1
            contribution = claim_map.get(zero_based_idx, {})
            judging_documents.append({
                "doc_index": doc_index,
                "doc_id": documents[zero_based_idx].get("id"),
                "claim": contribution.get("answer_or_claim", "NONE"),
                "contribution_type": contribution.get("contribution_type", "none"),
                "document_text": documents[zero_based_idx].get("context", ""),
            })
        return judging_documents

    def _build_judging_units(
        self,
        documents: List[Dict],
        cot_result: Dict,
        consolidation_step: Dict,
    ) -> List[Dict]:
        claim_map = self._build_doc_claim_map(cot_result)
        judging_units = []
        source_units = consolidation_step.get("abstracted_units", consolidation_step.get("canonical_units", []))
        for unit in source_units:
            if unit.get("gate_status") == "rejected_by_toxicity":
                continue
            unit_docs = []
            for doc_idx in unit.get("active_doc_indices", []):
                if doc_idx < 0 or doc_idx >= len(documents):
                    continue
                contribution = claim_map.get(doc_idx, {})
                unit_docs.append({
                    "doc_index": doc_idx + 1,
                    "doc_id": documents[doc_idx].get("id"),
                    "claim": contribution.get("answer_or_claim", unit.get("claim", "NONE")),
                    "contribution_type": contribution.get("contribution_type", "none"),
                    "document_text": documents[doc_idx].get("context", ""),
                })
            judging_units.append({
                "unit_id": unit["unit_id"],
                "claim": unit.get("claim", "NONE"),
                "member_claims": unit.get("member_claims", [unit.get("claim", "NONE")]),
                "doc_indices": [idx + 1 for idx in unit.get("active_doc_indices", [])],
                "docs": unit_docs,
            })
        return judging_units

    def _build_claim_only_consolidation_step(
        self,
        cot_result: Dict,
        toxicity_step: Optional[Dict] = None,
    ) -> Dict:
        viewpoint_toxicity = {
            item.get("viewpoint_index"): item
            for item in (toxicity_step or {}).get("viewpoint_toxicity", [])
        }
        rejected_raw_units = set((toxicity_step or {}).get("rejected_viewpoints", []))

        canonical_units = []
        raw_to_canonical = {}
        canonical_to_abstracted = {}
        unit_abstractions = []
        for canonical_id, raw_unit in enumerate(cot_result.get("support_units", [])):
            raw_unit_id = int(raw_unit.get("unit_id", canonical_id))
            claim = str(raw_unit.get("derived_answer", "")).strip() or "UNKNOWN"
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

            toxicity_item = viewpoint_toxicity.get(raw_unit_id, {})
            toxicity_score = _clamp_score(toxicity_item.get("toxicity_score"), default=0.5)
            gate_status = "rejected_by_toxicity" if raw_unit_id in rejected_raw_units else "survive"
            active_doc_indices = [] if gate_status == "rejected_by_toxicity" else doc_indices

            raw_to_canonical[raw_unit_id] = canonical_id
            canonical_to_abstracted[canonical_id] = canonical_id
            canonical_units.append({
                "unit_id": canonical_id,
                "claim": claim,
                "normalized_claim": _normalize_text_for_match(claim) or f"claim-only-{canonical_id}",
                "family_key": f"claim_only::{canonical_id}",
                "member_unit_ids": [canonical_id],
                "member_claims": [claim],
                "raw_unit_ids": [raw_unit_id],
                "all_doc_indices": doc_indices,
                "active_doc_indices": active_doc_indices,
                "toxicity_score_max": toxicity_score,
                "gate_status": gate_status,
                "abstraction_confidence": "ablation_disabled",
                "abstraction_rationale": "Ablation: answer-family merge and conflict graph construction are disabled.",
            })
            unit_abstractions.append({
                "unit_id": canonical_id,
                "base_answer": claim,
                "family_key": f"claim_only::{canonical_id}",
                "abstraction_confidence": "ablation_disabled",
                "rationale": "Ablation: this per-document claim remains an independent unit.",
            })

        return {
            "step": "support_unit_consolidation",
            "ablation_claim_only_no_conflict_graph": True,
            "canonical_units": canonical_units,
            "abstracted_units": canonical_units,
            "raw_to_canonical": raw_to_canonical,
            "canonical_to_abstracted": canonical_to_abstracted,
            "unit_abstractions": unit_abstractions,
            "pair_candidates": [],
            "surviving_unit_ids": [unit["unit_id"] for unit in canonical_units if unit["gate_status"] != "rejected_by_toxicity"],
            "rejected_unit_ids": [unit["unit_id"] for unit in canonical_units if unit["gate_status"] == "rejected_by_toxicity"],
        }

    def _build_default_toxicity_step(self) -> Dict:
        return {
            "step": "toxicity_coarse_filter",
            "toxicity_gate_threshold": float(self.config.get("toxicity_gate_threshold", 0.5)),
            "fallback_triggered": False,
            "surviving_viewpoints": [],
            "rejected_viewpoints": [],
            "viewpoint_toxicity": [],
            "viewpoint_analysis": [],
        }

    def _build_viewpoints_from_units(self, documents: List[Dict], cot_result: Dict) -> List[Dict]:
        claim_map = self._build_doc_claim_map(cot_result)
        viewpoints = []
        for unit in cot_result.get("support_units", []):
            all_doc_indices = []
            seen_doc_indices = set()
            for doc_idx in unit.get("docs", []):
                if not (0 <= doc_idx < len(documents)) or doc_idx in seen_doc_indices:
                    continue
                seen_doc_indices.add(doc_idx)
                all_doc_indices.append(doc_idx)
            if not all_doc_indices:
                continue

            # Toxicity scoring should only use direct answer-bearing evidence.
            scoring_docs = []
            scoring_doc_indices = []
            for doc_idx in all_doc_indices:
                contribution = claim_map.get(doc_idx, {})
                if contribution.get("contribution_type") != "final_answer":
                    continue
                doc_payload = dict(documents[doc_idx])
                doc_payload["contribution_type"] = contribution.get("contribution_type", "none")
                doc_payload["answer_or_claim"] = contribution.get("answer_or_claim", "NONE")
                scoring_docs.append(doc_payload)
                scoring_doc_indices.append(doc_idx)

            if not scoring_docs:
                continue
            viewpoints.append({
                "viewpoint_index": unit["unit_id"],
                "unit_id": unit["unit_id"],
                "claim": unit.get("derived_answer", "").strip(),
                "doc_indices": all_doc_indices,
                "docs": scoring_docs,
                "scoring_doc_indices": scoring_doc_indices,
            })
        viewpoints.sort(key=lambda item: item["viewpoint_index"])
        return viewpoints

    def _build_document_level_viewpoints(self, documents: List[Dict]) -> List[Dict]:
        viewpoints = []
        for doc_idx, document in enumerate(documents):
            doc_payload = dict(document)
            doc_payload["contribution_type"] = "document_context"
            doc_payload["answer_or_claim"] = "DOCUMENT_CONTEXT"
            viewpoints.append({
                "viewpoint_index": doc_idx,
                "unit_id": doc_idx,
                "claim": "DOCUMENT_CONTEXT",
                "doc_indices": [doc_idx],
                "docs": [doc_payload],
                "scoring_doc_indices": [doc_idx],
            })
        return viewpoints

    def _defend_doc_level_filtering(
        self,
        query: str,
        retrieved_documents: List[Dict],
        retrieve_fn: Callable,
    ) -> Dict:
        result = {
            "query": query,
            "original_doc_count": len(retrieved_documents),
            "defense_steps": [],
        }
        viewpoints = self._build_document_level_viewpoints(retrieved_documents)
        contradiction_result = {
            "has_contradiction": False,
            "doc_contributions": [
                {
                    "index": idx,
                    "contribution_type": "document_context",
                    "answer_or_claim": "DOCUMENT_CONTEXT",
                }
                for idx in range(len(retrieved_documents))
            ],
            "support_units": [],
            "contradictory_units": [],
            "overall_explanation": "Ablation: support-unit construction is disabled.",
            "ablation_mode": "doc_level_filtering",
        }
        result["defense_steps"].append({
            "step": "contradiction_detection",
            "ablation_doc_level_filtering": True,
            "has_contradiction": False,
            "raw_support_units": [],
            "raw_contradictory_units": [],
            "cot_analysis": contradiction_result,
        })
        result["raw_support_units"] = []
        result["raw_contradictory_units"] = []

        toxicity_step = self._coarse_filter_viewpoints_by_toxicity(query, viewpoints, retrieve_fn)
        result["defense_steps"].append(toxicity_step)

        surviving = set(toxicity_step.get("surviving_viewpoints", []))
        rejected = set(toxicity_step.get("rejected_viewpoints", []))
        toxicity_map = {
            item.get("viewpoint_index"): _clamp_score(item.get("toxicity_score"), default=0.5)
            for item in toxicity_step.get("viewpoint_toxicity", [])
        }

        final_documents = []
        retained_doc_ids = []
        filtered_doc_ids = []
        doc_scores = []
        for doc_idx, document in enumerate(retrieved_documents):
            if doc_idx in surviving:
                status = "retained_doc_level"
                final_documents.append(document)
                if document.get("id") is not None:
                    retained_doc_ids.append(document.get("id"))
            elif doc_idx in rejected:
                status = "filtered_doc_level"
                if document.get("id") is not None:
                    filtered_doc_ids.append(document.get("id"))
            else:
                status = "unmapped"
            toxicity_score = toxicity_map.get(doc_idx, 0.5)
            doc_scores.append({
                "doc_index": doc_idx + 1,
                "doc_id": document.get("id"),
                "claim": "DOCUMENT_CONTEXT",
                "IK_i": 0.0,
                "EV_i": 0.0,
                "R_i": max(0.0, 1.0 - toxicity_score),
                "toxicity_score": toxicity_score,
                "status": status,
                "rationale": "Ablation: document-level filtering without support-unit construction.",
                "unit_id": None,
            })

        if not final_documents:
            final_documents = list(retrieved_documents[:1])
            if final_documents[0].get("id") is not None and final_documents[0].get("id") not in retained_doc_ids:
                retained_doc_ids.append(final_documents[0].get("id"))

        result["typed_contradictions"] = []
        result["support_unit_consolidation"] = {
            "canonical_units": [],
            "abstracted_units": [],
            "raw_to_canonical": {},
            "canonical_to_abstracted": {},
            "unit_abstractions": [],
            "pair_candidates": [],
        }
        result["canonical_units"] = []
        result["abstracted_units"] = []
        result["internal_knowledge_note"] = {
            "tentative_answer": "I don't know",
            "key_facts": [],
            "confidence": "low",
        }
        result["query_form"] = infer_query_form(query)
        result["unit_reliability_scores"] = []
        result["doc_reliability_scores"] = doc_scores
        result["resolution_summary"] = {
            "query_form": result["query_form"],
            "factual_edges": [],
            "retained_unit_ids": [],
            "filtered_unit_ids": [],
            "retained_doc_ids": retained_doc_ids,
            "filtered_doc_ids": filtered_doc_ids,
            "unresolved_conflicts": [],
        }
        result["final_documents"] = final_documents
        result["defense_decision"] = "doc_level_filtering"
        result["extracted_claims"] = [
            {
                "viewpoint_index": idx + 1,
                "doc_index": idx + 1,
                "claim": "DOCUMENT_CONTEXT",
                "status": item["status"],
                "doc_ids": [retrieved_documents[idx].get("id")] if retrieved_documents[idx].get("id") is not None else [],
            }
            for idx, item in enumerate(doc_scores)
        ]
        return result

    def _collect_background_documents(
        self,
        documents: List[Dict],
        cot_result: Dict,
    ) -> List[Dict]:
        claim_map = self._build_doc_claim_map(cot_result)
        background_documents = []
        for doc_idx, document in enumerate(documents):
            contribution = claim_map.get(doc_idx, {})
            if contribution.get("contribution_type") != "background":
                continue
            background_documents.append(document)
        return background_documents

    def _build_single_viewpoint_background_result(
        self,
        query: str,
        documents: List[Dict],
        cot_result: Dict,
        viewpoints: List[Dict],
        result: Dict,
    ) -> Dict:
        background_documents = self._collect_background_documents(documents, cot_result)
        if len(viewpoints) != 1 or not background_documents:
            return result

        claim_map = self._build_doc_claim_map(cot_result)
        suppressed_unit = viewpoints[0]
        suppressed_doc_indices = set(suppressed_unit.get("doc_indices", []))
        retained_doc_ids = [doc.get("id") for doc in background_documents if doc.get("id") is not None]
        filtered_doc_ids = [
            documents[idx].get("id")
            for idx in suppressed_doc_indices
            if 0 <= idx < len(documents) and documents[idx].get("id") is not None
        ]

        result["defense_steps"].append({
            "step": "single_viewpoint_background_fallback",
            "reason": "Only one answer-bearing viewpoint was extracted and it is high-risk, so only background documents are forwarded to generation.",
            "suppressed_viewpoint_index": suppressed_unit.get("viewpoint_index"),
            "suppressed_claim": suppressed_unit.get("claim", ""),
            "retained_background_doc_ids": retained_doc_ids,
            "filtered_viewpoint_doc_ids": filtered_doc_ids,
        })
        result["typed_contradictions"] = []
        result["support_unit_consolidation"] = {
            "canonical_units": [{
                "unit_id": suppressed_unit.get("viewpoint_index"),
                "claim": suppressed_unit.get("claim", ""),
                "all_doc_indices": suppressed_unit.get("doc_indices", []),
                "active_doc_indices": [],
                "gate_status": "suppressed_single_viewpoint",
            }],
            "abstracted_units": [],
            "raw_to_canonical": {},
            "canonical_to_abstracted": {},
            "unit_abstractions": [],
        }
        result["canonical_units"] = result["support_unit_consolidation"]["canonical_units"]
        result["abstracted_units"] = []
        result["internal_knowledge_note"] = {
            "tentative_answer": "I don't know",
            "key_facts": [],
            "confidence": "low",
        }
        result["query_form"] = infer_query_form(query)
        result["unit_reliability_scores"] = [{
            "unit_id": suppressed_unit.get("viewpoint_index"),
            "claim": suppressed_unit.get("claim", ""),
            "doc_indices": [idx + 1 for idx in suppressed_unit.get("doc_indices", [])],
            "status": "suppressed_single_viewpoint",
            "rationale": "A single answer-bearing viewpoint is high-risk, so it is not forwarded directly; background evidence is used instead.",
        }]

        doc_scores = []
        for doc_idx, document in enumerate(documents):
            contribution = claim_map.get(doc_idx, {})
            if contribution.get("contribution_type") == "background":
                status = "retained_background_only"
                rationale = "Retained because only background evidence is forwarded when a lone viewpoint is extracted."
            elif doc_idx in suppressed_doc_indices:
                status = "suppressed_single_viewpoint"
                rationale = "Suppressed because it belongs to the only extracted answer-bearing viewpoint."
            else:
                status = "unmapped"
                rationale = "Document was not mapped into any canonical support unit."
            doc_scores.append({
                "doc_index": doc_idx + 1,
                "doc_id": document.get("id"),
                "claim": contribution.get("answer_or_claim", "NONE"),
                "IK_i": 0.0,
                "EV_i": 0.0,
                "R_i": 0.0,
                "status": status,
                "rationale": rationale,
                "unit_id": suppressed_unit.get("viewpoint_index") if doc_idx in suppressed_doc_indices else None,
            })

        result["doc_reliability_scores"] = doc_scores
        result["resolution_summary"] = {
            "query_form": result["query_form"],
            "factual_edges": [],
            "retained_unit_ids": [],
            "filtered_unit_ids": [suppressed_unit.get("viewpoint_index")],
            "retained_doc_ids": retained_doc_ids,
            "filtered_doc_ids": filtered_doc_ids,
            "unresolved_conflicts": [],
        }
        result["final_documents"] = background_documents
        result["defense_decision"] = "single_viewpoint_background_only"
        result["extracted_claims"] = self._collect_extracted_claims(
            documents,
            {
                "doc_projection": [
                    {
                        "doc_index": idx + 1,
                        "doc_id": doc.get("id"),
                        "claim": claim_map.get(idx, {}).get("answer_or_claim", "NONE"),
                        "gate_status": "retained_background_only" if claim_map.get(idx, {}).get("contribution_type") == "background" else "suppressed_single_viewpoint" if idx in suppressed_doc_indices else "unmapped",
                    }
                    for idx, doc in enumerate(documents)
                ]
            },
            {"doc_scores": doc_scores},
        )
        return result

    def _build_factual_viewpoint_graph(self, viewpoints: List[Dict], cot_result: Dict) -> nx.Graph:
        graph = nx.Graph()
        valid_indices = {vp["viewpoint_index"] for vp in viewpoints}
        for viewpoint in viewpoints:
            graph.add_node(
                viewpoint["viewpoint_index"],
                claim=viewpoint.get("claim", ""),
                doc_indices=viewpoint.get("doc_indices", []),
            )

        for conflict in cot_result.get("contradictory_units", []):
            unit1 = conflict.get("unit1")
            unit2 = conflict.get("unit2")
            if (
                conflict.get("type") == "factual"
                and unit1 in valid_indices
                and unit2 in valid_indices
                and unit1 != unit2
            ):
                graph.add_edge(
                    unit1,
                    unit2,
                    explanation=conflict.get("explanation", ""),
                    type=conflict.get("type"),
                )
        return graph

    def _dummy_retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        return []

    def _coarse_filter_viewpoints_by_toxicity(
        self,
        query: str,
        viewpoints: List[Dict],
        retrieve_fn: Callable,
    ) -> Dict:
        if self.config.get("ablation_disable_coarse_filter", False):
            viewpoint_toxicity = [
                {
                    "viewpoint_index": viewpoint["viewpoint_index"],
                    "claim": viewpoint.get("claim", ""),
                    "toxicity_score": 0.0,
                    "detector_status": "coarse_filter_ablation_disabled",
                    "status": "survive",
                }
                for viewpoint in viewpoints
            ]
            return {
                "step": "toxicity_coarse_filter",
                "toxicity_gate_threshold": float(self.config.get("toxicity_gate_threshold", 0.5)),
                "ablation_disabled": True,
                "fallback_triggered": False,
                "surviving_viewpoints": sorted(viewpoint["viewpoint_index"] for viewpoint in viewpoints),
                "rejected_viewpoints": [],
                "viewpoint_toxicity": sorted(viewpoint_toxicity, key=lambda item: item["viewpoint_index"]),
                "viewpoint_analysis": [
                    {
                        "viewpoint_index": viewpoint["viewpoint_index"],
                        "status": "coarse_filter_ablation_disabled",
                        "suspicion_score": 0.0,
                    }
                    for viewpoint in viewpoints
                ],
            }

        arbitration_result = self.arbitrate_viewpoints(query, viewpoints, retrieve_fn)
        viewpoint_analysis = arbitration_result.get("viewpoint_analysis", [])
        threshold = float(self.config.get("toxicity_gate_threshold", 0.5))
        fallback_keep_k = max(1, int(self.config.get("toxicity_fallback_keep_k", 2)))

        viewpoint_toxicity = []
        toxicity_by_index = {}
        surviving = []
        rejected = []

        for viewpoint in viewpoints:
            idx = viewpoint["viewpoint_index"]
            analysis = viewpoint_analysis[idx] if idx < len(viewpoint_analysis) else {}
            toxicity_score = _clamp_score(analysis.get("suspicion_score", 0.0), default=0.0)
            detector_status = analysis.get("status", "unknown")
            status = "survive"
            if toxicity_score > threshold:
                status = "rejected"
                rejected.append(idx)
            else:
                surviving.append(idx)
            toxicity_by_index[idx] = toxicity_score
            viewpoint_toxicity.append({
                "viewpoint_index": idx,
                "claim": viewpoint.get("claim", ""),
                "toxicity_score": toxicity_score,
                "detector_status": detector_status,
                "status": status,
            })

        fallback_triggered = False
        if not surviving and viewpoints:
            fallback_triggered = True
            ordered = sorted(viewpoints, key=lambda vp: (toxicity_by_index.get(vp["viewpoint_index"], 1.0), vp["viewpoint_index"]))
            rescued = [vp["viewpoint_index"] for vp in ordered[: min(fallback_keep_k, len(ordered))]]
            surviving = rescued
            rejected = [vp["viewpoint_index"] for vp in viewpoints if vp["viewpoint_index"] not in rescued]
            status_map = {item["viewpoint_index"]: item for item in viewpoint_toxicity}
            for idx in rescued:
                status_map[idx]["status"] = "rescued_by_fallback"

        return {
            "step": "toxicity_coarse_filter",
            "toxicity_gate_threshold": threshold,
            "fallback_triggered": fallback_triggered,
            "surviving_viewpoints": sorted(surviving),
            "rejected_viewpoints": sorted(rejected),
            "viewpoint_toxicity": sorted(viewpoint_toxicity, key=lambda item: item["viewpoint_index"]),
            "viewpoint_analysis": viewpoint_analysis,
        }

    def _build_retrieved_evidence_summary(
        self,
        documents: List[Dict],
        cot_result: Dict,
    ) -> str:
        contribution_map = {
            item["index"]: item
            for item in cot_result.get("doc_contributions", [])
        }
        unit_membership = {}
        for unit in cot_result.get("support_units", []):
            for doc_idx in unit.get("docs", []):
                unit_membership.setdefault(doc_idx, []).append(unit["unit_id"] + 1)

        lines = ["Document Overview:"]
        for idx, doc in enumerate(documents):
            item = contribution_map.get(idx, {})
            contribution_type = item.get("contribution_type", "none")
            answer_or_claim = item.get("answer_or_claim", "NONE")
            unit_text = unit_membership.get(idx, [])
            if unit_text:
                unit_desc = f"support_units={unit_text}"
            else:
                unit_desc = "support_units=[]"
            lines.append(
                f"- Doc {idx + 1}: {contribution_type} | claim={answer_or_claim} | {unit_desc}"
            )

        factual_conflicts = [
            conflict for conflict in cot_result.get("contradictory_units", [])
            if conflict.get("type") == "factual"
        ]
        if factual_conflicts:
            lines.append("Factual Conflict Summary:")
            for conflict in factual_conflicts:
                lines.append(
                    f"- Unit {conflict['unit1'] + 1} vs Unit {conflict['unit2'] + 1}: {conflict.get('explanation', '')}"
                )
        else:
            lines.append("Factual Conflict Summary: none")

        return "\n".join(lines)

    def _heuristic_risk_flags(
        self,
        query: str,
        viewpoint: Dict,
        cot_result: Dict,
        conflict_graph: Optional[nx.Graph] = None,
    ) -> List[str]:
        flags = set()
        docs = viewpoint.get("docs", [])
        if len(docs) == 1:
            flags.add("single_doc_only")

        if len(viewpoint.get("doc_indices", [])) == 1:
            flags.add("no_corroboration")

        if conflict_graph is not None and conflict_graph.degree(viewpoint["viewpoint_index"]) > 0:
            flags.add("conflicts_with_other_docs")

        normalized_query = _normalize_text_for_match(query)
        if normalized_query and docs:
            first_doc_text = _normalize_text_for_match(docs[0].get("context", ""))
            if first_doc_text.startswith(normalized_query):
                flags.add("query_copying")
                flags.add("synthetic_direct_answer")
            else:
                query_tokens = normalized_query.split()
                doc_prefix_tokens = first_doc_text.split()[: max(len(query_tokens) + 4, 12)]
                if query_tokens and len(set(query_tokens) & set(doc_prefix_tokens)) >= max(3, len(query_tokens) - 1):
                    flags.add("query_copying")

        contribution_map = {
            item["index"]: item
            for item in cot_result.get("doc_contributions", [])
        }
        answer_claims = []
        for doc_idx in viewpoint.get("doc_indices", []):
            answer_claims.append(contribution_map.get(doc_idx, {}).get("answer_or_claim", ""))
        if len(docs) == 1 and any(claim and claim.upper() not in INVALID_ANSWER_MARKERS for claim in answer_claims):
            flags.add("no_corroboration")

        return sorted(flags)

    def _parse_viewpoint_joint_scores(self, response: str, viewpoints: List[Dict]) -> Dict[int, Dict]:
        cleaned = _strip_code_fences(response)
        payload = None
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = _extract_top_level_json(cleaned)
            if payload is None:
                viewpoints_value = _extract_json_field(cleaned, "viewpoints")
                if isinstance(viewpoints_value, list):
                    payload = {"viewpoints": viewpoints_value}

        defaults = {
            viewpoint["viewpoint_index"]: {
                "viewpoint_index": viewpoint["viewpoint_index"],
                "claim": viewpoint.get("claim", ""),
                "support_score": 0.5,
                "evidence_score": 0.5,
                "risk_flags": [],
                "reason": "",
            }
            for viewpoint in viewpoints
        }

        if not payload:
            return defaults

        raw_viewpoints = payload.get("viewpoints", [])
        if not isinstance(raw_viewpoints, list):
            return defaults

        for item in raw_viewpoints:
            try:
                local_index = int(item.get("viewpoint_index")) - 1
            except Exception:
                continue
            if not (0 <= local_index < len(viewpoints)):
                continue
            global_index = viewpoints[local_index]["viewpoint_index"]
            raw_flags = item.get("risk_flags", [])
            if isinstance(raw_flags, str):
                raw_flags = [raw_flags]
            elif not isinstance(raw_flags, list):
                raw_flags = []
            defaults[global_index] = {
                "viewpoint_index": global_index,
                "claim": viewpoints[local_index].get("claim", ""),
                "support_score": _clamp_score(item.get("support_score"), default=0.5),
                "evidence_score": _clamp_score(item.get("evidence_score"), default=0.5),
                "risk_flags": sorted({
                    normalized
                    for normalized in (
                        _normalize_flag(flag)
                        for flag in raw_flags
                    )
                    if normalized is not None
                }),
                "reason": str(item.get("reason", "")).strip(),
            }

        return defaults

    def _score_viewpoints_jointly(
        self,
        query: str,
        viewpoints: List[Dict],
        documents: List[Dict],
        cot_result: Dict,
        conflict_graph: Optional[nx.Graph] = None,
    ) -> Dict:
        if not viewpoints:
            return {"step": "viewpoint_joint_scoring", "viewpoint_scores": []}

        evidence_summary = self._build_retrieved_evidence_summary(documents, cot_result)
        prompt = format_viewpoint_joint_scoring_prompt(query, viewpoints, evidence_summary)
        try:
            response = self.llm_model.query(prompt)
            parsed_scores = self._parse_viewpoint_joint_scores(response, viewpoints)
        except Exception as exc:
            print(f"Error in viewpoint joint scoring: {exc}")
            parsed_scores = self._parse_viewpoint_joint_scores("", viewpoints)

        for viewpoint in viewpoints:
            idx = viewpoint["viewpoint_index"]
            heuristic_flags = self._heuristic_risk_flags(query, viewpoint, cot_result, conflict_graph=conflict_graph)
            merged_flags = sorted(set(parsed_scores[idx].get("risk_flags", [])) | set(heuristic_flags))
            parsed_scores[idx]["risk_flags"] = merged_flags

        step_scores = sorted(parsed_scores.values(), key=lambda item: item["viewpoint_index"])
        return {
            "step": "viewpoint_joint_scoring",
            "other_evidence_summary": evidence_summary,
            "viewpoint_scores": step_scores,
        }

    def _resolve_viewpoint_conflicts(
        self,
        viewpoints: List[Dict],
        conflict_graph: nx.Graph,
        toxicity_step: Dict,
        scoring_step: Dict,
    ) -> Dict:
        support_weight = float(self.config.get("support_weight", 0.5))
        evidence_weight = float(self.config.get("evidence_weight", 0.5))
        total_weight = support_weight + evidence_weight
        if total_weight <= 0:
            support_weight = evidence_weight = 0.5
        else:
            support_weight /= total_weight
            evidence_weight /= total_weight

        resolution_margin = float(self.config.get("resolution_margin", 0.1))
        toxicity_map = {
            item["viewpoint_index"]: float(item.get("toxicity_score", 0.0))
            for item in toxicity_step.get("viewpoint_toxicity", [])
        }
        score_map = {
            item["viewpoint_index"]: item
            for item in scoring_step.get("viewpoint_scores", [])
        }

        all_scores = []
        for viewpoint in viewpoints:
            idx = viewpoint["viewpoint_index"]
            score_item = score_map.get(idx, {})
            support_score = _clamp_score(score_item.get("support_score"), default=0.5)
            evidence_score = _clamp_score(score_item.get("evidence_score"), default=0.5)
            risk_flags = sorted(set(score_item.get("risk_flags", [])))
            high_risk_combo = {
                "single_doc_only",
                "no_corroboration",
                "synthetic_direct_answer",
            }.issubset(set(risk_flags))
            risk_penalty = 0.0
            if high_risk_combo:
                risk_penalty += 0.2
            if "query_copying" in risk_flags:
                risk_penalty += 0.05
            if "conflicts_with_other_docs" in risk_flags:
                risk_penalty += 0.05
            risk_penalty = min(0.3, risk_penalty)
            final_score = support_weight * support_score + evidence_weight * evidence_score
            adjusted_score = max(0.0, final_score - risk_penalty)
            all_scores.append({
                "viewpoint_index": idx,
                "claim": viewpoint.get("claim", ""),
                "toxicity_score": toxicity_map.get(idx, 0.0),
                "support_score": support_score,
                "evidence_score": evidence_score,
                "risk_flags": risk_flags,
                "high_risk": high_risk_combo,
                "risk_penalty": risk_penalty,
                "final_score": final_score,
                "adjusted_score": adjusted_score,
            })

        score_lookup = {item["viewpoint_index"]: item for item in all_scores}
        surviving = set(toxicity_step.get("surviving_viewpoints", []))
        filtered = set(toxicity_step.get("rejected_viewpoints", []))
        retained = []
        unresolved_conflicts = []
        unresolved_keys = set()

        candidates = sorted(
            [idx for idx in surviving if idx in score_lookup],
            key=lambda idx: (-score_lookup[idx]["adjusted_score"], idx),
        )

        for idx in candidates:
            conflicting_retained = [other for other in retained if conflict_graph.has_edge(idx, other)]
            if not conflicting_retained:
                retained.append(idx)
                continue

            max_gap = max(
                score_lookup[other]["adjusted_score"] - score_lookup[idx]["adjusted_score"]
                for other in conflicting_retained
            )
            if max_gap > resolution_margin:
                filtered.add(idx)
                continue

            retained.append(idx)
            for other in conflicting_retained:
                key = tuple(sorted((idx, other)))
                if key in unresolved_keys:
                    continue
                unresolved_keys.add(key)
                edge_data = conflict_graph.get_edge_data(idx, other, default={})
                score_gap = abs(score_lookup[other]["adjusted_score"] - score_lookup[idx]["adjusted_score"])
                unresolved_conflicts.append({
                    "unit1": key[0],
                    "unit2": key[1],
                    "score_gap": score_gap,
                    "explanation": edge_data.get("explanation", "Retained both due to small score gap."),
                })

        retained_set = set(retained)
        filtered.update(idx for idx in surviving if idx not in retained_set)

        retained_viewpoints = [vp for vp in viewpoints if vp["viewpoint_index"] in retained_set]
        final_documents = []
        seen_doc_keys = set()
        for viewpoint in retained_viewpoints:
            for doc in viewpoint.get("docs", []):
                doc_key = _canonical_doc_key(doc)
                if doc_key in seen_doc_keys:
                    continue
                seen_doc_keys.add(doc_key)
                final_documents.append(doc)

        retained_doc_ids = []
        filtered_doc_ids = []
        retained_doc_keys = {_canonical_doc_key(doc) for doc in final_documents}

        for doc in final_documents:
            doc_id = doc.get("id")
            if doc_id is not None:
                retained_doc_ids.append(doc_id)

        for viewpoint in viewpoints:
            if viewpoint["viewpoint_index"] not in filtered:
                continue
            for doc in viewpoint.get("docs", []):
                doc_key = _canonical_doc_key(doc)
                if doc_key in retained_doc_keys:
                    continue
                doc_id = doc.get("id")
                if doc_id is not None and doc_id not in filtered_doc_ids:
                    filtered_doc_ids.append(doc_id)

        if not final_documents:
            decision = "no_retained_documents"
        elif len(retained) == 1 and score_lookup[retained[0]].get("high_risk", False):
            decision = "single_high_risk_viewpoint"
        elif conflict_graph.number_of_edges() == 0:
            decision = "resolved_single_viewpoint" if len(retained) == 1 else "no_factual_conflict"
        elif unresolved_conflicts:
            decision = "unresolved_factual_conflict"
        elif len(retained) == 1:
            decision = "resolved_single_viewpoint"
        else:
            decision = "resolved_multi_viewpoint"

        return {
            "step": "knowledge_aware_conflict_resolution",
            "weights": {
                "support": support_weight,
                "evidence": evidence_weight,
            },
            "resolution_margin": resolution_margin,
            "viewpoint_scores": sorted(all_scores, key=lambda item: item["viewpoint_index"]),
            "retained_viewpoints": sorted(retained),
            "filtered_viewpoints": sorted(filtered),
            "retained_doc_ids": retained_doc_ids,
            "filtered_doc_ids": filtered_doc_ids,
            "unresolved_conflicts": sorted(unresolved_conflicts, key=lambda item: (item["unit1"], item["unit2"])),
            "final_documents": final_documents,
            "decision": decision,
        }

    def _collect_extracted_claims(
        self,
        documents: List[Dict],
        doc_projection: Dict,
        resolution_step: Dict,
    ) -> List[Dict]:
        projection_map = {
            item["doc_index"]: item
            for item in doc_projection.get("doc_projection", [])
        }
        score_map = {
            item["doc_index"]: item
            for item in resolution_step.get("doc_scores", [])
        }

        claims = []
        for doc_index, document in enumerate(documents, start=1):
            projection = projection_map.get(doc_index, {})
            score_item = score_map.get(doc_index, {})
            status = score_item.get("status", projection.get("gate_status", "unknown"))
            if status in {"unmapped", "unknown"} and projection.get("gate_status") == "rejected_attack_instruction":
                status = "filtered_attack_instruction"
            claims.append({
                "viewpoint_index": doc_index,
                "doc_index": doc_index,
                "claim": score_item.get("claim", projection.get("claim", "NONE")),
                "status": status,
                "doc_ids": [document.get("id")] if document.get("id") is not None else [],
            })
        return claims

    def _dump_viewpoint_debug(self, query: str, viewpoints: List[Dict], resolution_step: Dict):
        if not self.debug_viewpoint_scores:
            return
        try:
            q_hash = str(hash(query))[:8]
            ts = time.strftime("%Y%m%d-%H%M%S")
            debug_path = os.path.join(self.debug_dir, f"{q_hash}_{ts}.json")
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "query": query,
                        "viewpoints": viewpoints,
                        "resolution": resolution_step,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=self._to_serializable,
                )
        except Exception as exc:
            print(f"[DEBUG] Failed to dump viewpoint debug: {exc}")

    def _visualize_viewpoint_graph(
        self,
        query: str,
        viewpoints: List[Dict],
        conflict_graph: nx.Graph,
        resolution_step: Dict,
    ):
        if not self.enable_visualization or not viewpoints:
            return
        try:
            os.makedirs(self.visualization_dir, exist_ok=True)
            retained = set(resolution_step.get("retained_viewpoints", []))
            filtered = set(resolution_step.get("filtered_viewpoints", []))
            viz_graph = nx.Graph()
            for viewpoint in viewpoints:
                idx = viewpoint["viewpoint_index"]
                status = "retained" if idx in retained else ("filtered" if idx in filtered else "unknown")
                color = "#2E8B57" if status == "retained" else "#FF8C00"
                viz_graph.add_node(
                    idx,
                    claim=viewpoint.get("claim", ""),
                    status=status,
                    color=color,
                    answer=viewpoint.get("claim", ""),
                    doc_count=len(viewpoint.get("docs", [])),
                )
            for u, v, data in conflict_graph.edges(data=True):
                viz_graph.add_edge(u, v, **data)

            query_hash = str(hash(query))[:8]
            viz_filename = f"viewpoint_graph_{query_hash}"
            self.visualizer.visualize_contradiction_graph(
                viz_graph,
                query=query,
                independent_set=set(retained),
                save_path=os.path.join(self.visualization_dir, f"{viz_filename}.png"),
                mode="viewpoint",
            )
            self.visualizer.create_interactive_summary(
                viz_graph,
                query=query,
                independent_set=set(retained),
                save_path=os.path.join(self.visualization_dir, f"{viz_filename}.html"),
            )
        except Exception as exc:
            print(f"Visualization generation failed: {exc}")
            traceback.print_exc()

    def _to_serializable(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, set):
            return list(obj)
        return obj

    def defend(
        self,
        query: str,
        retrieved_documents: List[Dict],
        retrieve_fn: Callable = None,
        traceback_retrieve_fn: Callable = None,
        traceback_seed_docs: Optional[List[Dict]] = None,
    ) -> Dict:
        result = {
            "query": query,
            "original_doc_count": len(retrieved_documents),
            "defense_steps": [],
        }

        if not retrieved_documents:
            result["final_documents"] = []
            result["defense_decision"] = "no_retained_documents"
            return result

        retrieve_fn = retrieve_fn or self._dummy_retrieve

        if self.config.get("ablation_doc_level_filtering", False):
            return self._defend_doc_level_filtering(query, retrieved_documents, retrieve_fn)

        use_per_document_extraction = (
            self.config.get("ablation_disable_global_semantic_conflict_cot", False)
            or self.config.get("ablation_claim_only_no_conflict_graph", False)
        )
        if use_per_document_extraction:
            contradiction_result = self.contradiction_detector.detect_contradictions_per_document_candidates(
                query,
                retrieved_documents,
            )
        else:
            contradiction_result = self.contradiction_detector.detect_contradictions_cot(query, retrieved_documents)
        result["defense_steps"].append({
            "step": "contradiction_detection",
            "ablation_disabled_global_semantic_conflict_cot": bool(
                self.config.get("ablation_disable_global_semantic_conflict_cot", False)
            ),
            "ablation_claim_only_no_conflict_graph": bool(
                self.config.get("ablation_claim_only_no_conflict_graph", False)
            ),
            "has_contradiction": contradiction_result["has_contradiction"],
            "raw_support_units": contradiction_result.get("support_units", []),
            "raw_contradictory_units": contradiction_result.get("contradictory_units", []),
            "cot_analysis": contradiction_result,
        })
        result["raw_support_units"] = contradiction_result.get("support_units", [])
        result["raw_contradictory_units"] = contradiction_result.get("contradictory_units", [])

        viewpoints = self._build_viewpoints_from_units(retrieved_documents, contradiction_result)
        if not viewpoints:
            if self.config.get("enable_poison_traceback_expansion", False) and traceback_seed_docs:
                traceback_step = self.poison_traceback_expander.expand(
                    query,
                    retrieved_documents,
                    [],
                    self._build_default_toxicity_step(),
                    {
                        "step": "support_unit_consolidation",
                        "canonical_units": [],
                        "abstracted_units": [],
                    },
                    traceback_retrieve_fn or retrieve_fn,
                    seed_override_docs=traceback_seed_docs,
                )
                if traceback_step:
                    result["defense_steps"].append(traceback_step)
            safe_documents, filtered_attack_doc_ids, _ = self._split_attack_instruction_documents(
                retrieved_documents,
                contradiction_result,
            )
            result["final_documents"] = safe_documents
            result["defense_decision"] = "no_support_units_fallback" if safe_documents else "no_retained_documents"
            result["extracted_claims"] = []
            result["typed_contradictions"] = []
            result["support_unit_consolidation"] = {
                "canonical_units": [],
                "abstracted_units": [],
                "raw_to_canonical": {},
                "canonical_to_abstracted": {},
                "unit_abstractions": [],
            }
            result["canonical_units"] = []
            result["abstracted_units"] = []
            result["internal_knowledge_note"] = {
                "tentative_answer": "I don't know",
                "key_facts": [],
                "confidence": "low",
            }
            result["query_form"] = infer_query_form(query)
            result["unit_reliability_scores"] = []
            result["doc_reliability_scores"] = []
            result["resolution_summary"] = {
                "retained_unit_ids": [],
                "filtered_unit_ids": [],
                "retained_doc_ids": [doc.get("id") for doc in safe_documents if doc.get("id") is not None],
                "filtered_doc_ids": filtered_attack_doc_ids,
                "unresolved_conflicts": [],
            }
            self._apply_attack_instruction_trace(result, retrieved_documents, contradiction_result)
            return result

        toxicity_step = self._coarse_filter_viewpoints_by_toxicity(query, viewpoints, retrieve_fn)

        # Single viewpoint high-risk fallback
        # If there's only 1 viewpoint, check if it's considered toxic (high risk)
        is_single_high_risk = False
        if len(viewpoints) == 1:
            vp_tox = toxicity_step.get("viewpoint_toxicity", [])
            if vp_tox and vp_tox[0].get("toxicity_score", 0.0) > toxicity_step.get("toxicity_gate_threshold", 0.5):
                is_single_high_risk = True

        if is_single_high_risk:
            single_viewpoint_background_result = self._build_single_viewpoint_background_result(
                query,
                retrieved_documents,
                contradiction_result,
                viewpoints,
                result,
            )
            if single_viewpoint_background_result.get("defense_decision") == "single_viewpoint_background_only":
                # Also include the toxicity step we ran, so it's in the trace
                single_viewpoint_background_result["defense_steps"].insert(-1, toxicity_step)
                self._apply_attack_instruction_trace(
                    single_viewpoint_background_result,
                    retrieved_documents,
                    contradiction_result,
                )
                return single_viewpoint_background_result

        result["defense_steps"].append(toxicity_step)
        doc_projection = self._project_toxicity_to_docs(
            retrieved_documents,
            viewpoints,
            toxicity_step,
            contradiction_result,
        )

        if self.config.get("ablation_claim_only_no_conflict_graph", False):
            consolidation_step = self._build_claim_only_consolidation_step(
                contradiction_result,
                toxicity_step=toxicity_step,
            )
        else:
            consolidation_step = self.support_unit_consolidator.consolidate_units(
                query,
                retrieved_documents,
                contradiction_result,
                toxicity_step=toxicity_step,
            )
        result["defense_steps"].append(consolidation_step)

        if self.config.get("enable_poison_traceback_expansion", False):
            traceback_step = self.poison_traceback_expander.expand(
                query,
                retrieved_documents,
                viewpoints,
                toxicity_step,
                consolidation_step,
                traceback_retrieve_fn or retrieve_fn,
                seed_override_docs=traceback_seed_docs,
            )
            if traceback_step:
                result["defense_steps"].append(traceback_step)

        pair_candidates = consolidation_step.get("pair_candidates", [])
        if self.config.get("ablation_disable_conflict_type_classification", False):
            typed_pairs = [
                {
                    "pair_id": pair["pair_id"],
                    "unit1": pair["unit1"],
                    "unit2": pair["unit2"],
                    "unit1_claim": pair.get("unit1_claim", "NONE"),
                    "unit2_claim": pair.get("unit2_claim", "NONE"),
                    "cot_explanation": pair.get("cot_explanation", ""),
                    "contradiction_type": "factual",
                    "rationale": "Ablation: contradiction type classifier disabled; all pair candidates are treated as factual.",
                }
                for pair in pair_candidates
            ]
            typed_pairs_step = {
                "step": "contradiction_type_classification",
                "ablation_disabled": True,
                "typed_pairs": typed_pairs,
                "factual_pairs": typed_pairs,
            }
        else:
            typed_pairs_step = self.contradiction_type_classifier.classify_pairs(query, pair_candidates)
        result["defense_steps"].append(typed_pairs_step)

        scoring_ablation_disabled = bool(self.config.get("ablation_disable_credibility_scoring", False))
        ik_ablation_disabled = scoring_ablation_disabled or bool(self.config.get("ablation_disable_internal_knowledge", False))
        ev_ablation_disabled = scoring_ablation_disabled or bool(self.config.get("ablation_disable_evidence_scoring", False))
        if ik_ablation_disabled:
            internal_note_step = {
                "step": "internal_knowledge_note",
                "ablation_disabled": True,
                "ablation_reason": "credibility_scoring_disabled" if scoring_ablation_disabled else "internal_knowledge_disabled",
                "note": {
                    "tentative_answer": "I don't know",
                    "key_facts": [],
                    "confidence": "low",
                },
                "raw_response": "",
            }
        else:
            internal_note_step = self.internal_knowledge_note_generator.generate(query)
        result["defense_steps"].append(internal_note_step)

        judging_units = self._build_judging_units(
            retrieved_documents,
            contradiction_result,
            consolidation_step,
        )
        if scoring_ablation_disabled:
            fallback_query_form = infer_query_form(query)
            judging_step = {
                "step": "evidence_aware_judging",
                "ablation_disabled": True,
                "ablation_reason": "credibility_scoring_disabled",
                "query_form": fallback_query_form,
                "units": [
                    {
                        "unit_id": unit["unit_id"],
                        "claim": unit.get("claim", "NONE"),
                        "doc_indices": unit.get("doc_indices", []),
                        "query_form": fallback_query_form,
                        "IK_u": 0.0,
                        "EV_raw": 0.0,
                        "EV_u": 0.0,
                        "query_restatement_risk": 0.0,
                        "linkage_gap_risk": 0.0,
                        "group_redundancy_risk": 0.0,
                        "support_pattern": "heuristic_rank_support",
                        "support_nature": "heuristic_rank_support",
                        "rationale": "Ablation: evidence-aware judging disabled; resolver uses retrieval-rank/support heuristic scores.",
                    }
                    for unit in judging_units
                ],
            }
        else:
            try:
                judging_step = self.evidence_aware_judge.judge_units(
                    query,
                    internal_note_step.get("note", {}),
                    judging_units,
                )
            except Exception as exc:
                fallback_query_form = infer_query_form(query)
                judging_step = {
                    "step": "evidence_aware_judging",
                    "query_form": fallback_query_form,
                    "units": [
                        {
                            "unit_id": unit["unit_id"],
                            "claim": unit.get("claim", "NONE"),
                            "doc_indices": unit.get("doc_indices", []),
                            "query_form": fallback_query_form,
                            "IK_u": 0.5,
                            "EV_raw": 0.5,
                            "EV_u": 0.5,
                            "query_restatement_risk": 0.5,
                            "linkage_gap_risk": 0.5,
                            "group_redundancy_risk": 0.0,
                            "support_pattern": "unclear",
                            "support_nature": "unclear",
                            "rationale": f"Fallback scores because evidence judging raised an exception: {exc}",
                        }
                        for unit in judging_units
                    ],
                }
        if ik_ablation_disabled and not scoring_ablation_disabled:
            judging_step["ablation_disable_internal_knowledge"] = True
            for unit_score in judging_step.get("units", []):
                unit_score["IK_u"] = 0.0
        if ev_ablation_disabled and not scoring_ablation_disabled:
            judging_step["ablation_disable_evidence_scoring"] = True
            for unit_score in judging_step.get("units", []):
                unit_score["EV_raw"] = 0.0
                unit_score["EV_u"] = 0.0
        result["defense_steps"].append(judging_step)

        resolution_step = self.knowledge_aware_resolver.resolve(
            query,
            retrieved_documents,
            consolidation_step.get("abstracted_units", consolidation_step.get("canonical_units", [])),
            judging_step.get("units", []),
            typed_pairs_step.get("factual_pairs", []),
            internal_note_step.get("note", {}),
        )
        result["defense_steps"].append({
            key: value for key, value in resolution_step.items() if key != "final_documents"
        })

        result["typed_contradictions"] = typed_pairs_step.get("typed_pairs", [])
        result["support_unit_consolidation"] = {
            "canonical_units": consolidation_step.get("canonical_units", []),
            "abstracted_units": consolidation_step.get("abstracted_units", []),
            "raw_to_canonical": consolidation_step.get("raw_to_canonical", {}),
            "canonical_to_abstracted": consolidation_step.get("canonical_to_abstracted", {}),
            "unit_abstractions": consolidation_step.get("unit_abstractions", []),
        }
        result["canonical_units"] = consolidation_step.get("canonical_units", [])
        result["abstracted_units"] = consolidation_step.get("abstracted_units", [])
        result["internal_knowledge_note"] = internal_note_step.get("note", {})
        result["query_form"] = judging_step.get("query_form", resolution_step.get("query_form"))
        result["unit_reliability_scores"] = resolution_step.get("unit_scores", [])
        result["doc_reliability_scores"] = resolution_step.get("doc_scores", [])
        result["resolution_summary"] = {
            "query_form": resolution_step.get("query_form"),
            "factual_edges": resolution_step.get("factual_edges", []),
            "retained_unit_ids": resolution_step.get("retained_unit_ids", []),
            "filtered_unit_ids": resolution_step.get("filtered_unit_ids", []),
            "retained_doc_ids": resolution_step.get("retained_doc_ids", []),
            "filtered_doc_ids": resolution_step.get("filtered_doc_ids", []),
            "unresolved_conflicts": resolution_step.get("unresolved_conflicts", []),
        }
        result["final_documents"] = resolution_step["final_documents"]
        result["defense_decision"] = resolution_step["decision"]
        result["extracted_claims"] = self._collect_extracted_claims(
            retrieved_documents,
            doc_projection,
            resolution_step,
        )
        self._apply_attack_instruction_trace(result, retrieved_documents, contradiction_result)

        self._dump_viewpoint_debug(query, viewpoints, resolution_step)
        return result
