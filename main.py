#!/usr/bin/env python
"""Command line interface for the SMS spam detector.

Examples
--------
    python main.py download
    python main.py train
    python main.py evaluate
    python main.py predict "Congratulations! You won a £1000 prize. Call now!"
    python main.py predict --file messages.csv
    python main.py api          # FastAPI service on :8000
    python main.py serve        # Streamlit dashboard
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import MODEL_FILE, PROJECT_ROOT, SPAM
from src.dataset import dataset_stats, download_dataset, ensure_dataset
from src.evaluate import compute_metrics, format_report, load_metrics
from src.model import SpamClassifier
from src.train import train


# --------------------------------------------------------------------- helpers
def _split_like_training(df: pd.DataFrame, metadata: dict):
    split = metadata.get("split", {})
    return train_test_split(
        df["message"],
        df["label"],
        test_size=split.get("test_size", 0.2),
        random_state=split.get("random_state", 42),
        stratify=df["label"],
    )


def _print_verdict(text: str, classifier: SpamClassifier, explain: bool, as_json: bool):
    probability = float(classifier.spam_probability([text])[0])
    label = SPAM if probability >= 0.5 else "ham"

    if as_json:
        print(
            json.dumps(
                {
                    "message": text,
                    "label": label,
                    "spam_probability": round(probability, 4),
                    "explanation": classifier.explain(text) if explain else [],
                },
                indent=2,
            )
        )
        return

    marker = "SPAM" if label == SPAM else "HAM "
    print(f"[{marker}] spam probability: {probability:.2%}")
    print(f'  "{text[:90]}{"..." if len(text) > 90 else ""}"')
    if explain:
        terms = classifier.explain(text)
        if terms:
            print("  strongest signals:")
            for item in terms:
                arrow = "spam" if item["weight"] > 0 else "ham "
                print(f"    {item['token']:<18} {item['weight']:+.3f}  ({arrow})")


# -------------------------------------------------------------------- commands
def cmd_download(args) -> int:
    df = download_dataset(force=args.force)
    stats = dataset_stats(df)
    print(
        f"Ready: {stats['rows']:,} messages "
        f"({stats['ham']:,} ham / {stats['spam']:,} spam, "
        f"{stats['spam_ratio']:.1%} spam)"
    )
    return 0


def cmd_train(args) -> int:
    result = train(
        test_size=args.test_size,
        seed=args.seed,
        cv_folds=args.cv,
        n_jobs=args.n_jobs,
        save=True,
    )
    print("\nCross-validation leaderboard")
    print("-" * 58)
    for row in result["leaderboard"]:
        print(f"  {row['model']:<22} f1(spam)={row['cv_f1_spam']:.4f}")
    return 0


def cmd_evaluate(args) -> int:
    classifier = SpamClassifier.load()
    df = ensure_dataset(verbose=False)
    _, X_test, _, y_test = _split_like_training(df, classifier.metadata)

    y_pred = classifier.predict(X_test)
    y_proba = classifier.spam_probability(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(format_report(metrics))
    if args.json:
        print(json.dumps(metrics, indent=2))
    return 0


def cmd_predict(args) -> int:
    classifier = SpamClassifier.load()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

        if path.suffix.lower() in {".txt", ".text"}:
            texts = [
                line.strip()
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
        else:
            frame = pd.read_csv(path)
            column = next(
                (
                    col
                    for col in ("message", "text", "sms", "v2")
                    if col in frame.columns
                ),
                frame.columns[0],
            )
            texts = frame[column].astype(str).tolist()

        probabilities = classifier.spam_probability(texts)
        output = pd.DataFrame(
            {
                "message": texts,
                "label": ["spam" if p >= 0.5 else "ham" for p in probabilities],
                "spam_probability": [round(float(p), 4) for p in probabilities],
            }
        )
        destination = path.with_name(path.stem + "_predictions.csv")
        output.to_csv(destination, index=False)
        print(output.head(20).to_string(index=False, max_colwidth=60))
        print(f"\n{len(output)} predictions saved -> {destination}")
        print(output["label"].value_counts().to_string())
        return 0

    if not args.text:
        print('Provide a message or use --file. Example: python main.py predict "You won £1000"')
        return 1

    for text in args.text:
        _print_verdict(text, classifier, args.explain, args.json)
    return 0


def cmd_info(args) -> int:
    metrics = load_metrics()
    if not metrics:
        print("No trained model yet. Run: python main.py train")
        return 1

    print(f"Model .............. {metrics['model']}")
    print(f"Trained at ......... {metrics['trained_at']}")
    print(f"Dataset ............ {metrics['dataset']['rows']:,} messages")
    print(f"CV F1 (spam) ....... {metrics['cv_f1_spam']:.4f}")
    print(f"Test accuracy ...... {metrics['test']['accuracy']:.4f}")
    print(f"Test F1 (spam) ..... {metrics['test']['spam']['f1']:.4f}")
    print(f"Test ROC AUC ....... {metrics['test'].get('roc_auc', float('nan')):.4f}")
    print(f"Best params ........ {metrics['best_params']}")
    print(f"Model file ......... {MODEL_FILE}")
    return 0


def cmd_serve(args) -> int:
    if not MODEL_FILE.exists():
        print("No trained model found - training one first.")
        train(n_jobs=1)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app.py"),
        "--server.port",
        str(args.port),
    ]
    print("Starting Streamlit dashboard ...")
    return subprocess.call(command, cwd=PROJECT_ROOT)


def cmd_api(args) -> int:
    """Run the FastAPI service with uvicorn."""
    if not MODEL_FILE.exists():
        print("No trained model found - training one first.")
        train(n_jobs=1)
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        command.append("--reload")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    return subprocess.call(command, cwd=PROJECT_ROOT)


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="SMS spam detection: train, evaluate and predict.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download the SMS Spam Collection")
    download.add_argument("--force", action="store_true", help="ignore the cached copy")
    download.set_defaults(func=cmd_download)

    train_cmd = sub.add_parser("train", help="train and save the best model")
    train_cmd.add_argument("--test-size", type=float, default=0.2)
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--cv", type=int, default=5)
    train_cmd.add_argument("--n-jobs", type=int, default=-1)
    train_cmd.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="re-score the saved model on the test split")
    evaluate.add_argument("--json", action="store_true", help="also dump raw metrics")
    evaluate.set_defaults(func=cmd_evaluate)

    predict = sub.add_parser("predict", help="classify one or more messages")
    predict.add_argument("text", nargs="*", help="message text")
    predict.add_argument("--file", help="CSV/TXT file of messages to label")
    predict.add_argument("--explain", action="store_true", help="show top contributing tokens")
    predict.add_argument("--json", action="store_true", help="machine-readable output")
    predict.set_defaults(func=cmd_predict)

    info = sub.add_parser("info", help="show saved model details")
    info.set_defaults(func=cmd_info)

    serve = sub.add_parser("serve", help="launch the Streamlit dashboard")
    serve.add_argument("--port", type=int, default=8501)
    serve.set_defaults(func=cmd_serve)

    api = sub.add_parser("api", help="run the FastAPI service")
    api.add_argument("--host", default="0.0.0.0")
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    api.set_defaults(func=cmd_api)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
