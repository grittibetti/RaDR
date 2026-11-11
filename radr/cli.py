from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from .preprocessing import BagPreprocessor, save_npz, _aggregate_target


def _split_csv(value: str | None):
    if value is None or str(value).strip() == "":
        return None
    return [part.strip() for part in str(value).split(",") if part.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess row-level CSV data into RaDR bag NPZ files.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--bag-id-col", required=True)
    parser.add_argument("--target-col", default=None)
    parser.add_argument(
        "--feature-cols",
        default=None,
        help="Comma-separated feature columns. Defaults to all non-ID/target/sort columns.",
    )
    parser.add_argument("--categorical-encoding", choices=["onehot", "ordinal", "numeric"], default="onehot")
    parser.add_argument("--categorical-cols", default=None)
    parser.add_argument("--handle-unknown", choices=["ignore", "error"], default="ignore")
    parser.add_argument("--sort-cols", default=None)
    parser.add_argument("--label-mode", choices=["bag", "aggregate"], default="bag")
    parser.add_argument("--aggregation", default="first")
    parser.add_argument("--scaler", choices=["standard", "minmax", "none"], default="standard")
    parser.add_argument("--missing", choices=["drop", "fill", "error"], default="drop")
    parser.add_argument("--fill-value", type=float, default=0.0)
    parser.add_argument("--min-bag-size", type=int, default=1)
    parser.add_argument("--max-bag-size", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    scaler = None if args.scaler == "none" else args.scaler
    prep = BagPreprocessor(
        bag_id_col=args.bag_id_col,
        feature_cols=_split_csv(args.feature_cols),
        target_col=args.target_col,
        task=args.task,
        label_mode=args.label_mode,
        aggregation=args.aggregation,
        sort_cols=_split_csv(args.sort_cols),
        scaler=scaler,
        categorical_encoding=args.categorical_encoding,
        categorical_cols=_split_csv(args.categorical_cols),
        handle_unknown=args.handle_unknown,
        missing=args.missing,
        fill_value=args.fill_value,
        min_bag_size=args.min_bag_size,
        max_bag_size=args.max_bag_size,
        random_state=args.random_state,
    )
    if df[args.bag_id_col].isna().any():
        raise ValueError("Bag IDs cannot be missing.")
    ids = df[args.bag_id_col].drop_duplicates().to_numpy()
    stratify = None
    if args.stratify:
        if args.task != "classification" or args.target_col is None:
            raise ValueError("--stratify requires classification and --target-col.")
        stratify = [
            _aggregate_target(
                df.loc[df[args.bag_id_col] == bag_id, args.target_col].to_numpy(),
                args.task,
                args.label_mode,
                args.aggregation,
            )
            for bag_id in ids
        ]
    train_ids, test_ids = train_test_split(
        ids, test_size=args.test_size, random_state=args.random_state, stratify=stratify
    )
    train_frame = df[df[args.bag_id_col].isin(train_ids)]
    test_frame = df[df[args.bag_id_col].isin(test_ids)]
    prep.fit(train_frame)
    counts = []
    for split, frame in [("train", train_frame), ("test", test_frame)]:
        bags, targets, metadata = prep.transform(frame, return_metadata=True)
        metadata.update(
            {
                "source_csv": str(Path(args.input_csv).resolve()),
                "split": split,
                "feature_names": prep.get_feature_names_out().tolist(),
            }
        )
        save_npz(
            output_dir / f"{split}_radr.npz", bags, targets, metadata=metadata, bag_ids=metadata["bag_ids"]
        )
        counts.append(len(bags))
    # Persist the training vocabulary and scaling statistics for inference.
    import joblib

    joblib.dump(prep, output_dir / "preprocessor.joblib")
    print(f"Saved {counts[0]} train bags and {counts[1]} test bags to {output_dir}")


if __name__ == "__main__":
    main()
