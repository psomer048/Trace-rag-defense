import sys, os
import json
import csv
import numpy as np
from collections import defaultdict
import random
try:
    import torch
except ImportError:
    torch = None
from .answer_utils import ensure_answer_list
from .dataset_profiles import (
    get_dataset_dir,
    get_dataset_profile,
    get_default_attack_manifest_path,
    get_default_query_manifest_path,
    get_corpus_path,
    get_qrels_path,
    get_queries_path,
    resolve_split,
)

model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

def contriever_get_emb(model, input):
    return model(**input)

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]


def _load_from_cache_first(loader, model_name, load_desc):
    try:
        print(f"[Model Load] Trying local cache for {load_desc}: {model_name}")
        return loader(model_name, local_files_only=True)
    except Exception as local_exc:
        print(
            f"[Model Load] Local cache unavailable for {load_desc}: {model_name}. "
            f"Falling back to default resolution."
        )
        return loader(model_name)


def load_models(model_code):
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if 'contriever' in model_code:
        from .contriever_src.contriever import Contriever
        from transformers import AutoTokenizer

        model_name = model_code_to_qmodel_name[model_code]
        model = _load_from_cache_first(Contriever.from_pretrained, model_name, "retriever model")
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = _load_from_cache_first(AutoTokenizer.from_pretrained, model_name, "retriever tokenizer")
        get_emb = contriever_get_emb
    elif 'ance' in model_code:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError
    
    return model, c_model, tokenizer, get_emb

def _read_jsonl(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_corpus(corpus_path):
    corpus = {}
    for item in _read_jsonl(corpus_path):
        doc_id = str(item.get("_id") or item.get("id"))
        if not doc_id:
            continue
        corpus[doc_id] = {
            "title": item.get("title", "") or "",
            "text": item.get("text", "") or "",
            "metadata": item.get("metadata", {}) or {},
        }
    return corpus


def _load_queries(queries_path):
    queries = {}
    for item in _read_jsonl(queries_path):
        query_id = str(item.get("_id") or item.get("id"))
        if not query_id:
            continue
        queries[query_id] = item.get("text") or item.get("question") or ""
    return queries


def _load_qrels(qrels_path):
    qrels = defaultdict(dict)
    if not os.path.exists(qrels_path):
        return qrels

    with open(qrels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            return qrels

        for row in reader:
            query_id = row.get("query-id") or row.get("query_id")
            corpus_id = row.get("corpus-id") or row.get("corpus_id")
            score = row.get("score", "1")
            if not query_id or not corpus_id:
                continue
            try:
                qrels[str(query_id)][str(corpus_id)] = int(float(score))
            except ValueError:
                qrels[str(query_id)][str(corpus_id)] = 1
    return qrels


def load_beir_datasets(dataset_name, split, data_path=None):
    profile = get_dataset_profile(dataset_name)
    split = resolve_split(dataset_name, split)
    dataset_dir = get_dataset_dir(dataset_name, data_path=data_path)

    if not os.path.exists(dataset_dir):
        if profile.beir_download_name:
            from beir import util

            datasets_root = os.path.dirname(dataset_dir)
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{profile.beir_download_name}.zip"
            dataset_dir = util.download_and_unzip(url, datasets_root)
        else:
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    queries_path = get_queries_path(dataset_name, data_path=dataset_dir)
    corpus_path = get_corpus_path(dataset_name, data_path=dataset_dir)
    qrels_path = get_qrels_path(dataset_name, split=split, data_path=dataset_dir)

    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"queries.jsonl not found for dataset '{dataset_name}' at {queries_path}")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"corpus.jsonl not found for dataset '{dataset_name}' at {corpus_path}")

    print(dataset_dir)
    corpus = _load_corpus(corpus_path)
    queries = _load_queries(queries_path)
    qrels = _load_qrels(qrels_path)
    return corpus, queries, qrels


def normalize_manifest_entry(query_id, payload, default_question=None):
    payload = dict(payload or {})
    normalized_id = str(payload.get("id") or query_id)
    question = payload.get("question") or payload.get("text") or default_question or ""
    correct_answers = ensure_answer_list(
        payload.get("correct_answers")
        or payload.get("possible_answers")
        or payload.get("answers")
        or []
    )
    canonical = payload.get("correct_answer") or payload.get("correct answer")
    if canonical:
        correct_answers = ensure_answer_list([canonical] + correct_answers)

    return {
        "id": normalized_id,
        "question": question,
        "correct_answer": str(canonical).strip() if canonical else (correct_answers[0] if correct_answers else ""),
        "correct_answers": correct_answers,
        "incorrect_answer": str(
            payload.get("incorrect_answer", payload.get("incorrect answer", ""))
        ).strip(),
        "adv_texts": list(payload.get("adv_texts") or []),
        "adv_prefixes": list(payload.get("adv_prefixes") or []),
        "adv_prefix_position": str(payload.get("adv_prefix_position") or "front").strip().lower(),
        "dataset": payload.get("dataset"),
        "metadata": payload.get("metadata", {}) or {},
    }


def load_query_manifest(dataset_name, manifest_path=None, queries=None):
    manifest_path = manifest_path or get_default_query_manifest_path(dataset_name)
    if os.path.exists(manifest_path):
        raw_manifest = load_json(manifest_path)
        return {
            str(query_id): normalize_manifest_entry(query_id, payload)
            for query_id, payload in raw_manifest.items()
        }

    fallback_attack_manifest = get_default_attack_manifest_path(dataset_name)
    if os.path.exists(fallback_attack_manifest):
        raw_manifest = load_json(fallback_attack_manifest)
        return {
            str(query_id): normalize_manifest_entry(query_id, payload)
            for query_id, payload in raw_manifest.items()
        }

    if queries is None:
        raise FileNotFoundError(
            f"No query manifest found for dataset '{dataset_name}' at {manifest_path}"
        )

    return {
        str(query_id): normalize_manifest_entry(query_id, {"question": question})
        for query_id, question in queries.items()
    }


def load_attack_manifest(dataset_name, manifest_path=None):
    manifest_path = manifest_path or get_default_attack_manifest_path(dataset_name)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"No attack manifest found for dataset '{dataset_name}' at {manifest_path}"
        )

    raw_manifest = load_json(manifest_path)
    return {
        str(query_id): normalize_manifest_entry(query_id, payload)
        for query_id, payload in raw_manifest.items()
    }


def merge_query_and_attack_manifests(query_manifest, attack_manifest=None):
    merged = {
        str(query_id): dict(payload)
        for query_id, payload in (query_manifest or {}).items()
    }
    for query_id, attack_payload in (attack_manifest or {}).items():
        base = dict(merged.get(str(query_id), {}))
        normalized_attack = normalize_manifest_entry(query_id, attack_payload)
        for key, value in normalized_attack.items():
            if key == "metadata":
                metadata = dict(base.get("metadata", {}) or {})
                metadata.update(value or {})
                base["metadata"] = metadata
            elif key == "correct_answers":
                if value:
                    base["correct_answers"] = ensure_answer_list(
                        value + base.get("correct_answers", [])
                    )
            elif value not in [None, "", []]:
                base[key] = value
        base["id"] = str(base.get("id") or query_id)
        base["question"] = base.get("question", "")
        base["correct_answer"] = base.get("correct_answer") or (
            base.get("correct_answers", [""])[0] if base.get("correct_answers") else ""
        )
        base["correct_answers"] = ensure_answer_list(base.get("correct_answers", []))
        base["adv_texts"] = list(base.get("adv_texts") or [])
        merged[str(query_id)] = base
    return merged

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_results(results, dir, file_name="debug"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f'results/query_results/{dir}'):
        os.makedirs(f'results/query_results/{dir}', exist_ok=True)
    with open(os.path.join(f'results/query_results/{dir}', f'{file_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)

def load_results(file_name):
    with open(os.path.join('results', file_name)) as file:
        results = json.load(file)
    return results

def save_json(results, file_path="debug.json"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def setup_seeds(seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)

def clean_str(s):
    try:
        s=str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s=s.strip()
    if len(s)>1 and s[-1] == ".":
        s=s[:-1]
    return s.lower()

def f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall arrays.
    
    Args:
    precision (np.array): A 2D array of precision values.
    recall (np.array): A 2D array of recall values.
    
    Returns:
    np.array: A 2D array of F1 scores.
    """
    f1_scores = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    
    return f1_scores
