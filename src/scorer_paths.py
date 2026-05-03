import os
from typing import List


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)

ACTIVE_POISON_SCORER_REL_PATH = "results/models/ACTIVE_POISON_SCORER.pkl"
ACTIVE_POISON_SCORER_TARGET_REL_PATH = "results/models/poison_rf_v2_bioasq_w134_clean_answerhit_gs4.pkl"
LEGACY_POISON_SCORER_REL_PATH = "src/poison_rf.pkl"
ARCHIVED_POISON_SCORER_DIR = "results/models/archive_non_gs4"


def _candidate_paths(path_value: str) -> List[str]:
    if not path_value:
        return []

    if os.path.isabs(path_value):
        return [path_value]

    return [
        os.path.abspath(path_value),
        os.path.join(PACKAGE_ROOT, path_value),
        os.path.join(REPO_ROOT, path_value),
    ]


def resolve_scorer_path(path_value: str) -> str:
    """
    Resolve a scorer model path robustly across different working directories.

    Preference order:
    1. the path as interpreted from the current working directory
    2. package-root-relative (poisonedrag_defense/)
    3. repo-root-relative

    If no candidate exists, return the package-root-relative version for the
    project defaults, otherwise fall back to the cwd-relative absolute path.
    """
    candidates = _candidate_paths(path_value)
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.realpath(candidate)

    if path_value == ACTIVE_POISON_SCORER_REL_PATH:
        fallback_candidates = _candidate_paths(ACTIVE_POISON_SCORER_TARGET_REL_PATH) + _candidate_paths(
            LEGACY_POISON_SCORER_REL_PATH
        )
        for candidate in fallback_candidates:
            if os.path.exists(candidate):
                return os.path.realpath(candidate)

    if not path_value:
        return path_value

    if os.path.isabs(path_value):
        return path_value

    package_relative = os.path.join(PACKAGE_ROOT, path_value)
    if path_value.startswith("poisonedrag_defense" + os.sep) or path_value.startswith("poisonedrag_defense/"):
        return os.path.join(REPO_ROOT, path_value)
    return package_relative


ACTIVE_POISON_SCORER_PATH = resolve_scorer_path(ACTIVE_POISON_SCORER_REL_PATH)
