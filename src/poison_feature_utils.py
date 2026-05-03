import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


LEGACY_FEATURE_NAMES: List[str] = [
    "group_size",
    "avg_grad_z",
    "avg_ppl",
    "echo_med",
    "radius",
    "max_sim",
    "susp_ratio",
]


V2_FEATURE_NAMES: List[str] = [
    "group_size",
    "avg_grad_z",
    "max_grad_z",
    "avg_ppl",
    "ppl_p90",
    "high_ppl_ratio",
    "avg_overlap_z",
    "high_overlap_ratio",
    "avg_query_cos",
    "max_query_cos",
    "query_cos_std",
    "avg_compression",
    "compression_std",
    "echo_med",
    "radius",
    "max_sim",
    "mean_nn_sim",
    "nn_sim_p75",
    "susp_ratio",
]


FEATURE_SETS: Dict[str, List[str]] = {
    "legacy": LEGACY_FEATURE_NAMES,
    "v2": V2_FEATURE_NAMES,
}


MODEL_METADATA_KEYS = {
    "model",
    "feature_names",
    "feature_set",
    "label_column",
    "threshold",
    "metrics",
    "notes",
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (list, tuple, dict, set)):
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


def _percentile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    if not values:
        return default
    return safe_float(np.percentile(np.asarray(values, dtype=float), q), default=default)


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return default
    return safe_float(np.mean(np.asarray(values, dtype=float)), default=default)


def _std(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return default
    return safe_float(np.std(np.asarray(values, dtype=float)), default=default)


def _max(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return default
    return safe_float(np.max(np.asarray(values, dtype=float)), default=default)


def _ratio(values: Sequence[float], predicate) -> float:
    if not values:
        return 0.0
    return safe_float(np.mean([1.0 if predicate(v) else 0.0 for v in values]), default=0.0)


def _stage1_view(group: Dict[str, Any]) -> Dict[str, float]:
    stage1 = group.get("stage1_features") or {}
    return {
        "echo_med": safe_float(stage1.get("echo_nn_med_ratio"), 0.0),
        "radius": safe_float(stage1.get("radius_mean"), 1.0),
        "max_sim": safe_float(stage1.get("max_nn_sim"), 0.0),
    }


def build_group_feature_map(group: Dict[str, Any]) -> Dict[str, float]:
    docs = list(group.get("docs") or [])
    stage1 = _stage1_view(group)

    grad_z = [safe_float(d.get("stage0_grad_z"), 0.0) for d in docs]
    ppl = [safe_float(d.get("stage2_ppl"), 0.0) for d in docs]
    overlap_z = [safe_float(d.get("stage2_ngram_overlap_q_z"), 0.0) for d in docs]
    query_cos = [safe_float(d.get("stage2_cos_sim_q"), 0.0) for d in docs]
    compression = [safe_float(d.get("stage2_compression_ratio"), 0.0) for d in docs]
    nn_sim = [safe_float(d.get("stage1_nn_sim"), 0.0) for d in docs]

    features = {
        "group_size": safe_float(len(docs), 0.0),
        "avg_grad_z": _mean(grad_z),
        "max_grad_z": _max(grad_z),
        "avg_ppl": _mean(ppl),
        "ppl_p90": _percentile(ppl, 90.0),
        "high_ppl_ratio": _ratio(ppl, lambda v: v > 150.0),
        "avg_overlap_z": _mean(overlap_z),
        "high_overlap_ratio": _ratio(overlap_z, lambda v: v > 1.0),
        "avg_query_cos": _mean(query_cos),
        "max_query_cos": _max(query_cos),
        "query_cos_std": _std(query_cos),
        "avg_compression": _mean(compression),
        "compression_std": _std(compression),
        "echo_med": stage1["echo_med"],
        "radius": stage1["radius"],
        "max_sim": stage1["max_sim"],
        "mean_nn_sim": _mean(nn_sim),
        "nn_sim_p75": _percentile(nn_sim, 75.0),
        "susp_ratio": safe_float(group.get("stage1_suspicious_ratio"), 0.0),
    }
    return features


def feature_vector_from_map(feature_map: Dict[str, float], feature_names: Iterable[str]) -> List[float]:
    return [safe_float(feature_map.get(name), 0.0) for name in feature_names]


def legacy_feature_vector(group: Dict[str, Any]) -> List[float]:
    feature_map = build_group_feature_map(group)
    return feature_vector_from_map(feature_map, LEGACY_FEATURE_NAMES)


def v2_feature_vector(group: Dict[str, Any]) -> List[float]:
    feature_map = build_group_feature_map(group)
    return feature_vector_from_map(feature_map, V2_FEATURE_NAMES)


def build_model_artifact(
    model: Any,
    feature_names: Sequence[str],
    feature_set: str,
    label_column: Optional[str] = None,
    threshold: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "model": model,
        "feature_names": list(feature_names),
        "feature_set": feature_set,
        "label_column": label_column,
        "threshold": threshold,
        "metrics": metrics or {},
        "notes": notes or {},
    }


def unpack_model_artifact(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and MODEL_METADATA_KEYS.intersection(obj.keys()):
        return {
            "model": obj.get("model"),
            "feature_names": list(obj.get("feature_names") or LEGACY_FEATURE_NAMES),
            "feature_set": obj.get("feature_set", "legacy"),
            "label_column": obj.get("label_column"),
            "threshold": obj.get("threshold"),
            "metrics": obj.get("metrics") or {},
            "notes": obj.get("notes") or {},
        }
    return {
        "model": obj,
        "feature_names": list(LEGACY_FEATURE_NAMES),
        "feature_set": "legacy",
        "label_column": None,
        "threshold": None,
        "metrics": {},
        "notes": {},
    }


def infer_poison_stats(docs: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    poison_count = 0
    total = 0
    for doc in docs or []:
        total += 1
        if bool(doc.get("is_poison")):
            poison_count += 1
    clean_count = max(total - poison_count, 0)
    poison_ratio = (poison_count / total) if total else 0.0
    return {
        "poison_doc_count": float(poison_count),
        "clean_doc_count": float(clean_count),
        "poison_ratio": float(poison_ratio),
        "label_any_poison": float(1 if poison_count > 0 else 0),
        "label_majority_poison": float(1 if total > 0 and poison_ratio >= 0.5 else 0),
        "label_all_poison": float(1 if total > 0 and poison_count == total else 0),
    }
