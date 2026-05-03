import re
from typing import Callable, Dict, List, Optional, Tuple


INVALID_CLAIMS = {"", "NONE", "UNKNOWN", "IDK", "N/A", "NULL"}
SYNTHETIC_ANSWER_PATTERNS = (
    "the answer is",
    "answer:",
    "correct answer",
    "when asked",
    "the correct response",
    "the correct answer",
)
PATCH_RISK_PATTERNS = (
    "according to the question",
    "for this question",
    "in response to the query",
    "contrary to popular belief",
    "it is often claimed",
    "some sources say",
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())).strip()


def _tokens(text: str) -> List[str]:
    return [token for token in _normalize_text(text).split() if token]


def _token_set(text: str) -> set:
    return set(_tokens(text))


def _jaccard(text_a: str, text_b: str) -> float:
    left = _token_set(text_a)
    right = _token_set(text_b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _doc_key(doc: Dict) -> Tuple[str, str]:
    doc_id = doc.get("id")
    if doc_id is not None:
        return ("id", str(doc_id))
    return ("context", doc.get("context", doc.get("text", "")))


def _preview(text: str, max_chars: int = 260) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def _is_valid_claim(claim: str) -> bool:
    return str(claim or "").strip().upper() not in INVALID_CLAIMS


class PoisonTracebackExpander:
    """
    Trace-only attack-conditioned re-retrieval.

    This class intentionally does not mutate documents, unit scores, final
    documents, or resolution outputs. It only records what a seed-conditioned
    re-retrieval would bring back from the caller-provided candidate pool.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enable_poison_traceback_expansion", False))
        self.pool_k = max(1, int(self.config.get("poison_traceback_pool_k", 100)))
        self.per_query_k = max(1, int(self.config.get("poison_traceback_per_query_k", 20)))
        self.max_seeds = max(1, int(self.config.get("poison_traceback_max_seeds", 3)))
        self.max_queries_per_seed = max(1, int(self.config.get("poison_traceback_max_queries_per_seed", 4)))
        self.min_seed_toxicity = float(self.config.get("poison_traceback_min_seed_toxicity", 0.5))
        self.trace_score_threshold = float(self.config.get("poison_traceback_score_threshold", 0.7))
        self.eval_mode = str(self.config.get("poison_traceback_eval_mode", "predicted_seed") or "predicted_seed")

    def expand(
        self,
        query: str,
        retrieved_documents: List[Dict],
        viewpoints: List[Dict],
        toxicity_step: Dict,
        consolidation_step: Dict,
        retrieve_fn: Callable,
        seed_override_docs: Optional[List[Dict]] = None,
    ) -> Dict:
        if not self.enabled:
            return {}

        step = {
            "step": "poison_traceback_expansion",
            "mode": "trace_only",
            "enabled": True,
            "mutates_final_context": False,
            "config": {
                "pool_k": self.pool_k,
                "per_query_k": self.per_query_k,
                "max_seeds": self.max_seeds,
                "max_queries_per_seed": self.max_queries_per_seed,
                "min_seed_toxicity": self.min_seed_toxicity,
                "trace_score_threshold": self.trace_score_threshold,
                "eval_mode": self.eval_mode,
            },
            "seeds": [],
            "retrieval_queries": [],
            "retrieved_candidates": [],
            "traced_poison_candidates": [],
            "summary": {
                "seed_count": 0,
                "retrieval_query_count": 0,
                "candidate_count": 0,
                "new_candidate_count": 0,
                "traced_count": 0,
            },
        }

        if retrieve_fn is None:
            step["summary"]["status"] = "no_retrieve_fn"
            return step

        seeds = self._select_seeds(
            retrieved_documents,
            viewpoints,
            toxicity_step,
            consolidation_step,
            seed_override_docs=seed_override_docs,
        )
        step["seeds"] = seeds
        step["summary"]["seed_count"] = len(seeds)
        if not seeds:
            step["summary"]["status"] = "no_suspicious_seed"
            return step

        initial_keys = {_doc_key(doc) for doc in retrieved_documents or []}
        candidate_map: Dict[Tuple[str, str], Dict] = {}

        for seed in seeds:
            for trace_query in self._build_trace_queries(query, seed):
                if len(seed["trace_queries"]) >= self.max_queries_per_seed:
                    break
                seed["trace_queries"].append(trace_query)
                step["retrieval_queries"].append({
                    "seed_id": seed["seed_id"],
                    "seed_type": seed["seed_type"],
                    "claim": seed["claim"],
                    "query": trace_query,
                })
                try:
                    candidates = retrieve_fn(trace_query, top_k=self.per_query_k)
                except Exception as exc:
                    step.setdefault("retrieval_errors", []).append({
                        "seed_id": seed["seed_id"],
                        "query": trace_query,
                        "error": str(exc),
                    })
                    continue

                for rank, candidate in enumerate(candidates or [], start=1):
                    key = _doc_key(candidate)
                    existing = candidate_map.get(key)
                    score = float(candidate.get("score", 0.0) or 0.0)
                    if existing is None:
                        candidate_map[key] = self._format_candidate(
                            candidate,
                            trace_query,
                            rank,
                            seed,
                            key not in initial_keys,
                            score,
                        )
                    else:
                        existing["retrieval_hits"].append({
                            "seed_id": seed["seed_id"],
                            "query": trace_query,
                            "rank": rank,
                            "score": score,
                        })
                        existing["best_retrieval_score"] = max(existing["best_retrieval_score"], score)

        candidates = list(candidate_map.values())
        for item in candidates:
            scoring = self._score_candidate(query, item, seeds)
            item.update(scoring)
            item.pop("context", None)

        candidates.sort(
            key=lambda item: (
                -float(item.get("poison_trace_score", 0.0)),
                not bool(item.get("is_new_candidate", False)),
                -float(item.get("best_retrieval_score", 0.0)),
                str(item.get("doc_id")),
            )
        )
        step["retrieved_candidates"] = candidates
        step["traced_poison_candidates"] = [
            item for item in candidates
            if item.get("is_new_candidate") and item.get("poison_trace_score", 0.0) >= self.trace_score_threshold
        ]
        for seed in step["seeds"]:
            seed.pop("seed_texts", None)
        step["summary"].update({
            "retrieval_query_count": len(step["retrieval_queries"]),
            "candidate_count": len(candidates),
            "new_candidate_count": sum(1 for item in candidates if item.get("is_new_candidate")),
            "traced_count": len(step["traced_poison_candidates"]),
            "status": "ok",
        })
        return step

    def _select_seeds(
        self,
        documents: List[Dict],
        viewpoints: List[Dict],
        toxicity_step: Dict,
        consolidation_step: Dict,
        seed_override_docs: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        if seed_override_docs:
            seeds = []
            for idx, doc in enumerate(seed_override_docs[: self.max_seeds]):
                seeds.append(self._seed_from_override_doc(idx, doc))
            return seeds

        viewpoint_lookup = {item.get("viewpoint_index"): item for item in viewpoints or []}
        toxicity_lookup = {
            item.get("viewpoint_index"): item
            for item in (toxicity_step or {}).get("viewpoint_toxicity", [])
        }
        rejected = set((toxicity_step or {}).get("rejected_viewpoints", []) or [])
        seeds: List[Dict] = []
        seen = set()

        for viewpoint_idx, tox_item in sorted(toxicity_lookup.items(), key=lambda pair: str(pair[0])):
            toxicity_score = float(tox_item.get("toxicity_score", 0.0) or 0.0)
            if viewpoint_idx not in rejected and toxicity_score < self.min_seed_toxicity:
                continue
            viewpoint = viewpoint_lookup.get(viewpoint_idx, {})
            claim = str(tox_item.get("claim") or viewpoint.get("claim") or "").strip()
            if not _is_valid_claim(claim):
                continue
            seed = self._seed_from_viewpoint(
                seed_id=f"vp:{viewpoint_idx}",
                seed_type="toxicity_viewpoint",
                viewpoint=viewpoint,
                claim=claim,
                toxicity_score=toxicity_score,
                documents=documents,
                reasons=["rejected_by_toxicity" if viewpoint_idx in rejected else "high_toxicity_score"],
            )
            key = (seed["seed_type"], seed["source_id"], _normalize_text(seed["claim"]))
            if key not in seen:
                seen.add(key)
                seeds.append(seed)

        source_units = consolidation_step.get("abstracted_units") or consolidation_step.get("canonical_units") or []
        for unit in source_units:
            toxicity_score = float(unit.get("toxicity_score_max", 0.0) or 0.0)
            gate_status = str(unit.get("gate_status", ""))
            if gate_status != "rejected_by_toxicity" and toxicity_score < self.min_seed_toxicity:
                continue
            claim = str(unit.get("claim", "")).strip()
            if not _is_valid_claim(claim):
                continue
            seed = self._seed_from_unit(
                seed_id=f"unit:{unit.get('unit_id')}",
                seed_type="consolidated_unit",
                unit=unit,
                claim=claim,
                toxicity_score=toxicity_score,
                documents=documents,
                reasons=[gate_status or "high_unit_toxicity"],
            )
            key = (seed["seed_type"], seed["source_id"], _normalize_text(seed["claim"]))
            if key not in seen:
                seen.add(key)
                seeds.append(seed)

        seeds.sort(key=lambda item: (-item.get("toxicity_score", 0.0), item["seed_id"]))
        return seeds[:self.max_seeds]

    def _seed_from_override_doc(self, seed_idx: int, doc: Dict) -> Dict:
        text = doc.get("context", doc.get("text", ""))
        claim = str(
            doc.get("answer_or_claim")
            or doc.get("claim")
            or doc.get("incorrect_answer")
            or ""
        ).strip()
        return {
            "seed_id": f"{self.eval_mode}:{seed_idx}",
            "seed_type": self.eval_mode,
            "source_id": doc.get("id"),
            "claim": claim,
            "toxicity_score": 1.0 if doc.get("is_poison") else 0.0,
            "doc_indices": [int(doc["initial_doc_index"]) + 1] if doc.get("initial_doc_index") is not None else [],
            "doc_ids": [doc.get("id")] if doc.get("id") is not None else [],
            "reasons": [self.eval_mode],
            "seed_doc_previews": [_preview(text)],
            "seed_texts": [text],
            "trace_queries": [],
        }

    def _seed_from_viewpoint(
        self,
        seed_id: str,
        seed_type: str,
        viewpoint: Dict,
        claim: str,
        toxicity_score: float,
        documents: List[Dict],
        reasons: List[str],
    ) -> Dict:
        doc_indices = list(viewpoint.get("doc_indices") or viewpoint.get("scoring_doc_indices") or [])
        seed_docs = self._seed_docs_from_indices(documents, doc_indices)
        return {
            "seed_id": seed_id,
            "seed_type": seed_type,
            "source_id": viewpoint.get("viewpoint_index"),
            "claim": claim,
            "toxicity_score": toxicity_score,
            "doc_indices": [idx + 1 for idx in doc_indices if isinstance(idx, int)],
            "doc_ids": [doc.get("id") for doc in seed_docs if doc.get("id") is not None],
            "reasons": reasons,
            "seed_doc_previews": [_preview(doc.get("context", doc.get("text", ""))) for doc in seed_docs[:3]],
            "seed_texts": [doc.get("context", doc.get("text", "")) for doc in seed_docs],
            "trace_queries": [],
        }

    def _seed_from_unit(
        self,
        seed_id: str,
        seed_type: str,
        unit: Dict,
        claim: str,
        toxicity_score: float,
        documents: List[Dict],
        reasons: List[str],
    ) -> Dict:
        doc_indices = list(unit.get("all_doc_indices") or unit.get("active_doc_indices") or [])
        seed_docs = self._seed_docs_from_indices(documents, doc_indices)
        return {
            "seed_id": seed_id,
            "seed_type": seed_type,
            "source_id": unit.get("unit_id"),
            "claim": claim,
            "toxicity_score": toxicity_score,
            "doc_indices": [idx + 1 for idx in doc_indices if isinstance(idx, int)],
            "doc_ids": [doc.get("id") for doc in seed_docs if doc.get("id") is not None],
            "reasons": reasons,
            "seed_doc_previews": [_preview(doc.get("context", doc.get("text", ""))) for doc in seed_docs[:3]],
            "seed_texts": [doc.get("context", doc.get("text", "")) for doc in seed_docs],
            "trace_queries": [],
        }

    def _seed_docs_from_indices(self, documents: List[Dict], doc_indices: List[int]) -> List[Dict]:
        seed_docs = []
        seen = set()
        for doc_idx in doc_indices:
            if not isinstance(doc_idx, int) or doc_idx < 0 or doc_idx >= len(documents) or doc_idx in seen:
                continue
            seen.add(doc_idx)
            seed_docs.append(documents[doc_idx])
        return seed_docs

    def _build_trace_queries(self, query: str, seed: Dict) -> List[str]:
        claim = str(seed.get("claim", "")).strip()
        queries = []
        candidates = [query, self._seed_pattern_query(seed, claim)]
        if claim:
            candidates.extend([f"{query} {claim}", self._claim_query(query, claim)])
        for candidate in candidates:
            cleaned = re.sub(r"\s+", " ", candidate or "").strip()
            if cleaned and cleaned not in queries:
                queries.append(cleaned)
        return queries

    def _claim_query(self, query: str, claim: str) -> str:
        query_tokens = [
            token for token in _tokens(query)
            if token not in {"what", "which", "who", "whom", "whose", "is", "are", "was", "were", "the", "a", "an"}
        ]
        claim_tokens = _tokens(claim)
        compact = " ".join((query_tokens[:8] + claim_tokens[:6])[:12])
        return compact or f"{query} {claim}"

    def _seed_pattern_query(self, seed: Dict, claim: str) -> str:
        seed_text = " ".join(seed.get("seed_texts") or [])
        tokens = [
            token for token in _tokens(seed_text)
            if len(token) > 3 and token not in {"this", "that", "with", "from", "have", "about", "question"}
        ]
        unique = []
        for token in tokens:
            if token not in unique:
                unique.append(token)
            if len(unique) >= 8:
                break
        claim_tokens = [token for token in _tokens(claim) if token not in unique]
        return " ".join((unique + claim_tokens[:4])[:12])

    def _format_candidate(
        self,
        candidate: Dict,
        trace_query: str,
        rank: int,
        seed: Dict,
        is_new_candidate: bool,
        score: float,
    ) -> Dict:
        text = candidate.get("context", candidate.get("text", ""))
        return {
            "doc_id": candidate.get("id"),
            "source": candidate.get("source", "traceback_retrieve_fn"),
            "is_new_candidate": bool(is_new_candidate),
            "is_poison_label": bool(candidate.get("is_poison", False)),
            "best_retrieval_score": score,
            "context_preview": _preview(text),
            "context": text,
            "retrieval_hits": [{
                "seed_id": seed["seed_id"],
                "query": trace_query,
                "rank": rank,
                "score": score,
            }],
        }

    def _score_candidate(self, query: str, candidate: Dict, seeds: List[Dict]) -> Dict:
        text = candidate.get("context", "")
        norm_text = _normalize_text(text)
        query_copying = self._query_copying_risk(query, text)
        synthetic_risk = 1.0 if any(pattern in norm_text for pattern in SYNTHETIC_ANSWER_PATTERNS) else 0.0
        patch_risk = 1.0 if any(pattern in norm_text for pattern in PATCH_RISK_PATTERNS) else 0.0

        best_seed = None
        best_claim_support = 0.0
        best_seed_similarity = 0.0
        for seed in seeds:
            claim_support = self._claim_support_score(seed.get("claim", ""), text)
            seed_similarity = max((_jaccard(seed_text, text) for seed_text in seed.get("seed_texts", [])), default=0.0)
            combined = claim_support + seed_similarity
            if best_seed is None or combined > (best_claim_support + best_seed_similarity):
                best_seed = seed
                best_claim_support = claim_support
                best_seed_similarity = seed_similarity

        trace_score = (
            0.30 * best_claim_support
            + 0.25 * min(best_seed_similarity / 0.35, 1.0)
            + 0.20 * query_copying
            + 0.15 * synthetic_risk
            + 0.10 * patch_risk
        )
        reasons = []
        if best_claim_support >= 0.8:
            reasons.append("supports_seed_claim")
        if best_seed_similarity >= 0.25:
            reasons.append("similar_to_seed_text")
        if query_copying >= 0.7:
            reasons.append("query_copying_risk")
        if synthetic_risk:
            reasons.append("synthetic_direct_answer_risk")
        if patch_risk:
            reasons.append("prefix_or_patch_risk")

        return {
            "matched_seed_id": best_seed.get("seed_id") if best_seed else None,
            "matched_claim": best_seed.get("claim") if best_seed else "",
            "attack_claim_support": round(best_claim_support, 6),
            "seed_similarity": round(best_seed_similarity, 6),
            "query_copying_risk": round(query_copying, 6),
            "synthetic_direct_answer_risk": round(synthetic_risk, 6),
            "prefix_patch_risk": round(patch_risk, 6),
            "poison_trace_score": round(max(0.0, min(1.0, trace_score)), 6),
            "trace_reasons": reasons,
        }

    def _claim_support_score(self, claim: str, text: str) -> float:
        claim_norm = _normalize_text(claim)
        text_norm = _normalize_text(text)
        if not claim_norm or not text_norm:
            return 0.0
        if claim_norm in text_norm:
            return 1.0
        claim_tokens = [token for token in claim_norm.split() if len(token) > 2]
        if not claim_tokens:
            return 0.0
        text_tokens = set(text_norm.split())
        return len([token for token in claim_tokens if token in text_tokens]) / max(1, len(claim_tokens))

    def _query_copying_risk(self, query: str, text: str) -> float:
        query_tokens = _tokens(query)
        if not query_tokens:
            return 0.0
        prefix_tokens = _tokens(text)[: max(12, len(query_tokens) + 6)]
        if not prefix_tokens:
            return 0.0
        prefix_text = " ".join(prefix_tokens)
        query_text = " ".join(query_tokens)
        if prefix_text.startswith(query_text):
            return 1.0
        overlap = len(set(query_tokens) & set(prefix_tokens)) / max(1, len(set(query_tokens)))
        return max(0.0, min(1.0, overlap))
