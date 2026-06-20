from pathlib import Path
import pickle
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------------------------
# Try importing clean_text from utils/preprocess.py
# If not available, use fallback cleaner
# ------------------------------------------------
try:
    from utils.preprocess import clean_text
except Exception:
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ROLE_DATA_PATH = DATA_DIR / "resumes.csv"
DETECTOR_DATA_PATH = DATA_DIR / "resume_detector.csv"

ROLE_MODEL_PATH = MODELS_DIR / "role_model.pkl"
ROLE_VECTORIZER_PATH = MODELS_DIR / "role_vectorizer.pkl"
DETECTOR_MODEL_PATH = MODELS_DIR / "detector_model.pkl"
DETECTOR_VECTORIZER_PATH = MODELS_DIR / "detector_vectorizer.pkl"


# -----------------------------
# Possible column names
# -----------------------------
ROLE_TEXT_CANDIDATES = ["resume_text", "Resume_str", "Text", "text"]
ROLE_LABEL_CANDIDATES = ["category", "Category", "label"]

DET_TEXT_CANDIDATES = ["text", "resume_text", "Text", "Resume_str"]
DET_LABEL_CANDIDATES = ["is_resume", "label", "category", "Category"]


# -----------------------------
# Helpers
# -----------------------------
def find_column(df, candidates):
    cleaned = {col.strip().lower(): col for col in df.columns}
    for c in candidates:
        key = c.strip().lower()
        if key in cleaned:
            return cleaned[key]

    raise ValueError(
        f"Could not find any of these columns: {candidates}. Found: {list(df.columns)}"
    )


def normalize_detector_labels(series):
    def convert(val):
        s = str(val).strip().lower()

        if s in ["1", "true", "yes", "resume"]:
            return "Resume"
        if s in ["0", "false", "no", "not resume", "not_resume"]:
            return "Not Resume"

        return str(val).strip()

    return series.apply(convert)


def read_csv_safely(path):
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except Exception as e:
            last_error = e

    raise last_error


# -----------------------------
# Dataset loaders
# -----------------------------
def load_role_dataset():
    if not ROLE_DATA_PATH.exists():
        raise FileNotFoundError(f"Role dataset not found: {ROLE_DATA_PATH}")

    df = read_csv_safely(ROLE_DATA_PATH)
    df.columns = df.columns.str.strip()

    text_col = find_column(df, ROLE_TEXT_CANDIDATES)
    label_col = find_column(df, ROLE_LABEL_CANDIDATES)

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    df.dropna(inplace=True)
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df["cleaned"] = df["text"].apply(clean_text)
    df = df[df["cleaned"].str.len() > 0].copy()

    if df.empty:
        raise ValueError("Role dataset is empty after cleaning.")

    return df


def build_bootstrap_resume_detector(role_df):
    not_resume_examples = [
        "invoice total amount tax gst bill due date quantity item payment received",
        "bank statement account number debit credit balance transaction date branch",
        "electricity bill meter number bill amount due date consumer address payment",
        "railway ticket passenger seat coach departure arrival pnr booking platform",
        "meeting agenda minutes attendees action items deadline follow up discussion",
        "sample pdf document table of contents chapter introduction appendix page",
        "purchase order vendor item rate quantity subtotal shipping invoice number",
        "medical prescription patient dosage tablet diagnosis doctor hospital clinic",
        "school report card student marks subject attendance grade examination result",
        "newspaper article headline reporter publication date editorial opinion column",
        "court notice case number hearing date petitioner respondent legal document",
        "restaurant bill order amount taxes service charge payment method receipt",
    ]

    resumes = role_df[["text"]].copy()
    resumes["label"] = "Resume"

    non_resumes = pd.DataFrame(
        {
            "text": not_resume_examples,
            "label": "Not Resume",
        }
    )

    detector_df = pd.concat([resumes, non_resumes], ignore_index=True)
    return detector_df


def load_detector_dataset(role_df):
    if DETECTOR_DATA_PATH.exists() and DETECTOR_DATA_PATH.stat().st_size > 0:
        df = read_csv_safely(DETECTOR_DATA_PATH)
        df.columns = df.columns.str.strip()

        text_col = find_column(df, DET_TEXT_CANDIDATES)
        label_col = find_column(df, DET_LABEL_CANDIDATES)

        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "label"]

        df.dropna(inplace=True)
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = normalize_detector_labels(df["label"])
    else:
        print("resume_detector.csv not found or empty. Using bootstrap detector dataset.")
        df = build_bootstrap_resume_detector(role_df)

    df["cleaned"] = df["text"].apply(clean_text)
    df = df[df["cleaned"].str.len() > 0].copy()

    if df.empty:
        raise ValueError("Detector dataset is empty after cleaning.")

    return df


# -----------------------------
# Model training
# -----------------------------
def train_classifier(df, purpose_name):
    if df["label"].nunique() < 2:
        raise ValueError(
            f"{purpose_name} needs at least 2 classes, but found: {df['label'].unique()}"
        )

    label_counts = df["label"].value_counts()
    use_stratify = label_counts.min() >= 2

    X = df["cleaned"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None,
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=2.0,
        solver="lbfgs"
    )

    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    print(f"\n===== {purpose_name} evaluation =====")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, zero_division=0))

    return model, vectorizer


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Loading role dataset...")
    role_df = load_role_dataset()
    print(f"Loaded {len(role_df)} role training rows.")

    print("Loading detector dataset...")
    detector_df = load_detector_dataset(role_df)
    print(f"Loaded {len(detector_df)} detector training rows.")

    print("\nTraining role classifier...")
    role_model, role_vectorizer = train_classifier(role_df, "Role classifier")

    print("\nTraining resume detector...")
    detector_model, detector_vectorizer = train_classifier(
        detector_df, "Resume detector"
    )

    save_pickle(role_model, ROLE_MODEL_PATH)
    save_pickle(role_vectorizer, ROLE_VECTORIZER_PATH)
    save_pickle(detector_model, DETECTOR_MODEL_PATH)
    save_pickle(detector_vectorizer, DETECTOR_VECTORIZER_PATH)

    print("\nTraining completed successfully!")
    print(
        f"Saved: {ROLE_MODEL_PATH.name}, "
        f"{ROLE_VECTORIZER_PATH.name}, "
        f"{DETECTOR_MODEL_PATH.name}, "
        f"{DETECTOR_VECTORIZER_PATH.name}"
    )