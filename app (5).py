"""Streamlit dashboard for the SMS Spam Detector.

Talks to the FastAPI service (``SPAM_API_URL``, default http://localhost:8000)
and falls back to the bundled model when the API is not running, so the demo
works in a notebook, on a laptop, or against a deployed container.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.client import DEFAULT_API_URL, APIError, SpamAPIClient
from src.config import MODEL_FILE
from src.model import SpamClassifier
from src.train import train

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="wide")

API_URL = os.getenv("SPAM_API_URL", DEFAULT_API_URL)

EXAMPLES = [
    "Congratulations! You have won a £1000 cash prize. Call 0800 123 4567 to claim now!",
    "Hey, are we still meeting for lunch tomorrow at 1?",
    "URGENT: Your account will be suspended. Verify your details at http://secure-login.example",
    "Ok, see you at the station in 20 minutes.",
]


# --------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner=False)
def get_client(url: str) -> SpamAPIClient:
    return SpamAPIClient(url, timeout=15.0)


@st.cache_resource(show_spinner=False)
def get_local_model() -> SpamClassifier | None:
    try:
        return SpamClassifier.load()
    except FileNotFoundError:
        return None


def api_info() -> dict | None:
    """Model info from the API, or None when it is unreachable."""
    try:
        return get_client(API_URL).model_info()
    except APIError:
        return None


def local_info(classifier: SpamClassifier) -> dict:
    metadata = classifier.metadata
    test = metadata.get("test", {})
    return {
        "name": metadata.get("model", "unknown"),
        "version": classifier.version,
        "trained_at": metadata.get("trained_at"),
        "dataset": metadata.get("dataset", {}),
        "metrics": {
            key: test[key] for key in ("accuracy", "macro_f1", "roc_auc") if key in test
        },
        "spam_metrics": test.get("spam", {}),
        "best_params": metadata.get("best_params", {}),
        "cv_f1_spam": metadata.get("cv_f1_spam"),
        "leaderboard": metadata.get("leaderboard", []),
    }


def classify(text: str, threshold: float) -> tuple[dict, str]:
    """Return ``(prediction, source)`` trying the API and falling back locally."""
    try:
        return get_client(API_URL).predict(text, explain=True, threshold=threshold), "api"
    except APIError:
        classifier = get_local_model()
        if classifier is None:
            raise
        probability = float(classifier.spam_probability([text])[0])
        return (
            {
                "label": "spam" if probability >= threshold else "ham",
                "spam_probability": probability,
                "confidence": max(probability, 1 - probability),
                "threshold": threshold,
                "explanation": classifier.explain(text),
                "model": {"name": classifier.name, "version": classifier.version},
                "latency_ms": None,
            },
            "local",
        )


def classify_many(messages: list[str], threshold: float) -> tuple[list[dict], str]:
    try:
        payload = get_client(API_URL).predict_batch(messages, threshold=threshold)
        return payload["results"], "api"
    except APIError:
        classifier = get_local_model()
        if classifier is None:
            raise
        probabilities = classifier.spam_probability(messages)
        return (
            [
                {
                    "message": message,
                    "label": "spam" if probability >= threshold else "ham",
                    "spam_probability": float(probability),
                }
                for message, probability in zip(messages, probabilities, strict=True)
            ],
            "local",
        )


def source_badge(source: str) -> None:
    if source == "api":
        st.caption(f"Served by the FastAPI backend at `{API_URL}`.")
    else:
        st.caption(
            f"API unreachable at `{API_URL}` - using the local model in `{MODEL_FILE.name}`."
        )


# ----------------------------------------------------------------------- tabs
def render_result(result: dict, threshold: float, source: str) -> None:
    probability = float(result["spam_probability"])
    if result["label"] == "spam":
        st.error(
            f"### Spam detected\nThis message is spam with {probability:.1%} confidence."
        )
    else:
        st.success(
            f"### Looks legitimate\nHam with {1 - probability:.1%} confidence "
            f"(spam probability {probability:.1%})."
        )

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(f"Spam probability: **{probability:.1%}**")

    columns = st.columns(3)
    columns[0].metric("Confidence", f"{float(result['confidence']):.1%}")
    columns[1].metric("Threshold", f"{threshold:.2f}")
    latency = result.get("latency_ms")
    columns[2].metric("Latency", f"{latency:.1f} ms" if latency else "local")

    terms = result.get("explanation") or []
    if terms:
        with st.expander("Why? Top contributing tokens", expanded=True):
            frame = pd.DataFrame(terms).rename(columns={"weight": "contribution"})
            frame["pushes towards"] = frame["contribution"].apply(
                lambda value: "spam" if value > 0 else "ham"
            )
            st.dataframe(frame, hide_index=True, width="stretch")

    source_badge(source)


def single_message_tab(threshold: float) -> None:
    if "message_input" not in st.session_state:
        st.session_state.message_input = EXAMPLES[0]

    def use_example(example: str) -> None:
        # on_click runs before the rerun, so changing widget state is allowed.
        st.session_state.message_input = example

    st.write("Paste an SMS message below.")
    text = st.text_area("Message", key="message_input", height=140)

    columns = st.columns(len(EXAMPLES))
    for index, (column, example) in enumerate(zip(columns, EXAMPLES, strict=False)):
        column.button(
            "Spam example" if index == 0 else "Try this",
            key=f"example_{index}",
            on_click=use_example,
            args=(example,),
            width="stretch",
        )

    if st.button("Analyse message", type="primary", width="stretch"):
        if not text.strip():
            st.warning("Please enter a message first.")
            return
        try:
            result, source = classify(text.strip(), threshold)
        except APIError as exc:
            st.error(f"Classification failed: {exc}")
            return
        render_result(result, threshold, source)


def batch_tab(threshold: float) -> None:
    st.write(
        "Upload a CSV with a `message` (or `text`) column, or a `.txt` file with "
        "one message per line."
    )
    upload = st.file_uploader("Messages", type=["csv", "txt"])
    if upload is None:
        return

    if upload.name.lower().endswith(".txt"):
        texts = [
            line.strip()
            for line in upload.read().decode("utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        original = None
    else:
        original = pd.read_csv(upload)
        column = next(
            (name for name in ("message", "text", "sms", "v2") if name in original.columns),
            original.columns[0],
        )
        texts = original[column].astype(str).tolist()

    if not texts:
        st.warning("No messages found in that file.")
        return

    try:
        results, source = classify_many(texts, threshold)
    except APIError as exc:
        st.error(f"Batch classification failed: {exc}")
        return

    frame = pd.DataFrame(results)
    spam_count = int((frame["label"] == "spam").sum())
    left, middle, right = st.columns(3)
    left.metric("Messages", f"{len(frame):,}")
    middle.metric("Flagged spam", f"{spam_count:,}")
    right.metric("Spam rate", f"{spam_count / len(frame):.1%}")

    st.dataframe(frame, hide_index=True, width="stretch")

    if original is not None and len(original) == len(frame):
        combined = pd.concat(
            [original.reset_index(drop=True), frame.drop(columns=["message"])], axis=1
        )
    else:
        combined = frame
    st.download_button(
        "Download predictions as CSV",
        combined.to_csv(index=False).encode("utf-8"),
        file_name="sms_predictions.csv",
        mime="text/csv",
        width="stretch",
    )
    source_badge(source)


def model_tab(info: dict | None) -> None:
    if info is None:
        st.warning(
            "No model available: the API is unreachable and there is no local artifact. "
            "Run `python main.py train` or start the API."
        )
        return

    metrics = info.get("metrics", {})
    spam = info.get("spam_metrics", {})
    dataset = info.get("dataset", {})

    left, middle, right, last = st.columns(4)
    left.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    middle.metric("Spam precision", f"{spam.get('precision', 0):.2%}")
    right.metric("Spam recall", f"{spam.get('recall', 0):.2%}")
    last.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")

    st.write(
        f"**Model:** `{info.get('name', 'n/a')}` &nbsp;|&nbsp; "
        f"**Version:** `{info.get('version', 'n/a')}` &nbsp;|&nbsp; "
        f"**Trained:** {info.get('trained_at', 'n/a')} &nbsp;|&nbsp; "
        f"**Dataset:** {dataset.get('rows', 0):,} messages ({dataset.get('spam', 0):,} spam)"
    )
    if info.get("best_params"):
        st.caption(f"Hyper-parameters: `{info['best_params']}`")

    leaderboard = info.get("leaderboard") or []
    if leaderboard:
        st.subheader("Cross-validation leaderboard")
        st.dataframe(
            pd.DataFrame(leaderboard).rename(
                columns={"cv_f1_spam": "CV F1 (spam)", "best_params": "best params"}
            ),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Retrain locally")
    st.caption(
        "Retrains the model on disk (5-fold grid search, a few seconds). If the API is "
        "running it keeps serving the old artifact until you call `POST /admin/reload`."
    )
    if st.button("Retrain now"):
        with st.spinner("Training ..."):
            train(n_jobs=1, verbose=False)
        get_local_model.clear()
        st.success("Model retrained and saved to models/.")
        st.rerun()


# ----------------------------------------------------------------------- page
def main() -> None:
    info = api_info()
    classifier = get_local_model()
    local = local_info(classifier) if classifier else None
    effective_info = info or local

    st.title("📩 SMS Spam Detector")
    st.caption(
        "TF-IDF + linear classifier trained on the SMS Spam Collection. "
        "The dashboard prefers the FastAPI service and falls back to the local model."
    )

    with st.sidebar:
        st.header("Settings")
        threshold = st.slider(
            "Spam threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.5,
            step=0.05,
            help="Messages above this spam probability are flagged.",
        )
        st.divider()
        st.subheader("Backend")
        if info:
            st.success("API online")
            st.caption(f"`{API_URL}`")
        else:
            st.warning("API offline - using local model")
            st.caption(f"`{API_URL}`")
        if effective_info:
            st.write(f"**{effective_info['name'].replace('_', ' ').title()}**")
            st.write(f"Version: `{effective_info['version']}`")
        st.divider()
        st.caption(
            "Lower the threshold to catch more spam at the cost of false positives."
        )

    single, batch, about = st.tabs(["Single message", "Batch file", "Model"])
    with single:
        if effective_info is None:
            st.warning("No trained model found. Run `python main.py train` first.")
        else:
            single_message_tab(threshold)
    with batch:
        if effective_info is None:
            st.warning("No trained model found. Run `python main.py train` first.")
        else:
            batch_tab(threshold)
    with about:
        model_tab(effective_info)

    st.divider()
    st.caption(
        "Not a substitute for a real SMS gateway filter. "
        f"Docs: {API_URL}/docs . Metrics: {API_URL}/metrics"
    )


if __name__ == "__main__":
    main()
