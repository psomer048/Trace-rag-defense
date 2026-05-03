import argparse
import os
from typing import List

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.poison_feature_utils import FEATURE_SETS, build_model_artifact


DEFAULT_DATA_FILE = "poison_training_data.csv"
DEFAULT_MODEL_FILE = "poison_rf.pkl"


def assign_soft_label_from_group_size(group_size: float) -> float:
    if group_size >= 5:
        return 1.0
    if group_size == 4:
        return 0.85
    if group_size == 3:
        return 0.65
    if group_size == 2:
        return 0.45
    return 0.25


def resolve_feature_names(schema: str, custom_columns: str = "") -> List[str]:
    if custom_columns:
        return [col.strip() for col in custom_columns.split(",") if col.strip()]
    if schema not in FEATURE_SETS:
        raise ValueError(f"Unknown schema: {schema}")
    return list(FEATURE_SETS[schema])


def train(args: argparse.Namespace):
    try:
        df = pd.read_csv(args.data_file)
    except FileNotFoundError:
        print(f"Error: {args.data_file} not found.")
        return

    feature_names = resolve_feature_names(args.schema, args.feature_columns)
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    if args.mode == "soft":
        if args.soft_target_column and args.soft_target_column in df.columns:
            y = df[args.soft_target_column]
            print(f"Using existing soft target column: {args.soft_target_column}")
        elif args.soft_target_source == "poison_ratio":
            if "poison_ratio" not in df.columns:
                raise ValueError("soft_target_source=poison_ratio requires a poison_ratio column")
            y = df["poison_ratio"]
            print("Using poison_ratio as soft target.")
        else:
            if "group_size" not in df.columns:
                raise ValueError("soft_target_source=group_size requires a group_size column")
            y = df["group_size"].apply(assign_soft_label_from_group_size)
            print("Using legacy group_size-derived soft targets.")
        eval_label = (y > 0).astype(int)
    else:
        if args.label_column not in df.columns:
            raise ValueError(f"Label column not found: {args.label_column}")
        y = df[args.label_column]
        eval_label = y
        print(f"Using binary label column: {args.label_column}")

    X = df[feature_names]
    if len(df) < 10:
        print("Not enough data to train. Need at least 10 rows.")
        return

    sample_weight = None
    if args.sample_weight_column:
        if args.sample_weight_column not in df.columns:
            raise ValueError(f"Sample weight column not found: {args.sample_weight_column}")
        sample_weight = df[args.sample_weight_column]
    groups = None
    if args.group_by_column:
        if args.group_by_column not in df.columns:
            raise ValueError(f"Group-by column not found: {args.group_by_column}")
        groups = df[args.group_by_column].astype(str)

    print(f"Loaded {len(df)} rows from {args.data_file}")
    print(f"Schema: {args.schema} ({len(feature_names)} features)")
    print("Feature columns:")
    print(", ".join(feature_names))
    if groups is not None:
        print(f"Group split column: {args.group_by_column} ({groups.nunique()} groups)")

    test_size = float(args.test_size)
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=args.random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        label_train, label_test = eval_label.iloc[train_idx], eval_label.iloc[test_idx]
        if sample_weight is not None:
            w_train, w_test = sample_weight.iloc[train_idx], sample_weight.iloc[test_idx]
        else:
            w_train = None
    else:
        stratify = eval_label if args.mode == "binary" else None
        split_result = train_test_split(
            X,
            y,
            eval_label,
            *( [sample_weight] if sample_weight is not None else [] ),
            test_size=test_size,
            random_state=args.random_state,
            stratify=stratify,
        )
        if sample_weight is not None:
            X_train, X_test, y_train, y_test, label_train, label_test, w_train, w_test = split_result
        else:
            X_train, X_test, y_train, y_test, label_train, label_test = split_result
            w_train = None

    if args.mode == "soft":
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )

    if w_train is not None:
        model.fit(X_train, y_train, sample_weight=w_train)
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {}
    print("\nModel Evaluation:")
    if args.mode == "soft":
        mse = mean_squared_error(y_test, y_pred)
        threshold = float(args.eval_threshold)
        y_pred_bin = (y_pred >= threshold).astype(int)
        acc = accuracy_score(label_test, y_pred_bin)
        report = classification_report(label_test, y_pred_bin, zero_division=0)
        metrics.update({"mse": float(mse), "binary_accuracy": float(acc), "eval_threshold": threshold})
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Binary Inference Accuracy (Threshold={threshold:.2f}): {acc:.4f}")
        print(report)
    else:
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        metrics.update({"accuracy": float(acc)})
        print(f"Accuracy: {acc:.4f}")
        print(report)

    importances = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nFeature Importances:")
    print(importances.to_string(index=False))

    artifact = build_model_artifact(
        model=model,
        feature_names=feature_names,
        feature_set=args.schema,
        label_column=args.label_column if args.mode == "binary" else args.soft_target_column,
        threshold=(float(args.eval_threshold) if args.mode == "soft" else None),
        metrics=metrics,
        notes={
            "mode": args.mode,
            "data_file": args.data_file,
            "soft_target_source": args.soft_target_source if args.mode == "soft" else None,
            "sample_weight_column": args.sample_weight_column or None,
            "group_by_column": args.group_by_column or None,
        },
    )
    model_dir = os.path.dirname(os.path.abspath(args.model_file))
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    joblib.dump(artifact, args.model_file)
    print(f"\nModel saved to {args.model_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the poison scorer")
    parser.add_argument("--data-file", type=str, default=DEFAULT_DATA_FILE)
    parser.add_argument("--model-file", type=str, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--schema", type=str, choices=sorted(FEATURE_SETS.keys()), default="legacy")
    parser.add_argument("--feature-columns", type=str, default="", help="Comma-separated override for feature columns")
    parser.add_argument("--mode", type=str, choices=["binary", "soft"], default="binary")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--soft-target-column", type=str, default="")
    parser.add_argument("--soft-target-source", type=str, choices=["group_size", "poison_ratio"], default="group_size")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--eval-threshold", type=float, default=0.3)
    parser.add_argument("--sample-weight-column", type=str, default="")
    parser.add_argument("--group-by-column", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
