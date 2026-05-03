import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    default_split: str = "test"
    supports_gt_injection: bool = True
    has_gold_evidence: str = "yes"
    answer_mode: str = "single"
    corpus_mode: str = "local"
    corpus_source: Optional[str] = None
    manifest_required: bool = False
    beir_download_name: Optional[str] = None
    forced_split: Optional[str] = None


DATASET_PROFILES = {
    "nq": DatasetProfile(
        name="nq",
        default_split="test",
        supports_gt_injection=True,
        has_gold_evidence="yes",
        answer_mode="single",
        corpus_mode="local",
        beir_download_name="nq",
    ),
    "open_nq": DatasetProfile(
        name="open_nq",
        default_split="test",
        supports_gt_injection=False,
        has_gold_evidence="yes",
        answer_mode="multi_alias",
        corpus_mode="local",
        manifest_required=True,
    ),
    "msmarco": DatasetProfile(
        name="msmarco",
        default_split="train",
        supports_gt_injection=True,
        has_gold_evidence="yes",
        answer_mode="single",
        corpus_mode="local",
        beir_download_name="msmarco",
        forced_split="train",
    ),
    "hotpotqa": DatasetProfile(
        name="hotpotqa",
        default_split="test",
        supports_gt_injection=True,
        has_gold_evidence="yes",
        answer_mode="single",
        corpus_mode="local",
        beir_download_name="hotpotqa",
    ),
    "popqa": DatasetProfile(
        name="popqa",
        default_split="test",
        supports_gt_injection=False,
        has_gold_evidence="no",
        answer_mode="multi_alias",
        corpus_mode="shared",
        corpus_source="nq",
        manifest_required=True,
    ),
    "bioasq_factoid": DatasetProfile(
        name="bioasq_factoid",
        default_split="train",
        supports_gt_injection=False,
        has_gold_evidence="yes",
        answer_mode="multi_alias",
        corpus_mode="local",
        manifest_required=True,
    ),
    "asqa_qa_pairs": DatasetProfile(
        name="asqa_qa_pairs",
        default_split="dev",
        supports_gt_injection=False,
        has_gold_evidence="partial",
        answer_mode="multi_alias",
        corpus_mode="local",
        manifest_required=True,
    ),
    "asqa_gtr_top100_qa_pairs": DatasetProfile(
        name="asqa_gtr_top100_qa_pairs",
        default_split="dev",
        supports_gt_injection=False,
        has_gold_evidence="partial",
        answer_mode="multi_alias",
        corpus_mode="local",
        manifest_required=True,
    ),
    "asqa_dpr_top100_qa_pairs": DatasetProfile(
        name="asqa_dpr_top100_qa_pairs",
        default_split="dev",
        supports_gt_injection=False,
        has_gold_evidence="partial",
        answer_mode="multi_alias",
        corpus_mode="local",
        manifest_required=True,
    ),
    "asqa_oracle_qa_pairs": DatasetProfile(
        name="asqa_oracle_qa_pairs",
        default_split="dev",
        supports_gt_injection=False,
        has_gold_evidence="partial",
        answer_mode="multi_alias",
        corpus_mode="local",
        manifest_required=True,
    ),
}


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_datasets_root() -> str:
    return os.path.join(get_project_root(), "datasets")


def get_results_root() -> str:
    return os.path.join(get_project_root(), "results")


def get_dataset_profile(dataset_name: str) -> DatasetProfile:
    if dataset_name in DATASET_PROFILES:
        return DATASET_PROFILES[dataset_name]

    dataset_dir = os.path.join(get_datasets_root(), dataset_name)
    if os.path.isdir(dataset_dir):
        return DatasetProfile(name=dataset_name)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def resolve_split(dataset_name: str, split: Optional[str] = None) -> str:
    profile = get_dataset_profile(dataset_name)
    if profile.forced_split:
        return profile.forced_split
    return split or profile.default_split


def get_dataset_dir(dataset_name: str, data_path: Optional[str] = None) -> str:
    if data_path:
        return os.path.abspath(data_path)
    return os.path.join(get_datasets_root(), dataset_name)


def get_corpus_path(dataset_name: str, data_path: Optional[str] = None) -> str:
    profile = get_dataset_profile(dataset_name)
    dataset_dir = get_dataset_dir(dataset_name, data_path=data_path)
    local_corpus = os.path.join(dataset_dir, "corpus.jsonl")
    if os.path.exists(local_corpus):
        return local_corpus

    if profile.corpus_source:
        source_dir = get_dataset_dir(profile.corpus_source)
        return os.path.join(source_dir, "corpus.jsonl")

    return local_corpus


def get_queries_path(dataset_name: str, data_path: Optional[str] = None) -> str:
    dataset_dir = get_dataset_dir(dataset_name, data_path=data_path)
    return os.path.join(dataset_dir, "queries.jsonl")


def get_qrels_path(dataset_name: str, split: Optional[str] = None, data_path: Optional[str] = None) -> str:
    dataset_dir = get_dataset_dir(dataset_name, data_path=data_path)
    effective_split = resolve_split(dataset_name, split)
    return os.path.join(dataset_dir, "qrels", f"{effective_split}.tsv")


def get_default_query_manifest_path(dataset_name: str) -> str:
    return os.path.join(get_results_root(), "query_manifests", f"{dataset_name}.json")


def get_default_attack_manifest_path(dataset_name: str) -> str:
    return os.path.join(get_results_root(), "adv_targeted_results", f"{dataset_name}.json")
