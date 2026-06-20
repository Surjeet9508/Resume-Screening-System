from pathlib import Path
import pickle
import math
import pandas as pd
import streamlit as st

from utils.preprocess import clean_text
from utils.extractor import extract_pdf, extract_docx

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

ROLE_MODEL_PATH = MODELS_DIR / "role_model.pkl"
ROLE_VECTORIZER_PATH = MODELS_DIR / "role_vectorizer.pkl"
DETECTOR_MODEL_PATH = MODELS_DIR / "detector_model.pkl"
DETECTOR_VECTORIZER_PATH = MODELS_DIR / "detector_vectorizer.pkl"

# -----------------------------
# Settings
# -----------------------------
RESUME_THRESHOLD = 0.55
APPLICABLE_THRESHOLD = 60.0
SELECTED_THRESHOLD = 80.0

# No score boost
SCORE_MULTIPLIER = 1.00
MAX_DISPLAY_SCORE = 99.00

# -----------------------------
# Session state for clearing uploads
# -----------------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Resume Screening System", page_icon="📄", layout="wide")

st.title("📄 Resume Screening System")
st.write(
    "Upload one or more PDF/DOCX files. The system classifies each file as Resume or Not Resume. "
    "If it is a resume, the system predicts the job role and score."
)

# -----------------------------
# Check model files
# -----------------------------
required_files = [
    ROLE_MODEL_PATH,
    ROLE_VECTORIZER_PATH,
    DETECTOR_MODEL_PATH,
    DETECTOR_VECTORIZER_PATH,
]

missing = [str(p.name) for p in required_files if not p.exists()]
if missing:
    st.error(
        "Model files not found: " + ", ".join(missing) + ". "
        "Run this command first:\n\npython train_model.py"
    )
    st.stop()

# -----------------------------
# Load models
# -----------------------------
with open(ROLE_MODEL_PATH, "rb") as f:
    role_model = pickle.load(f)

with open(ROLE_VECTORIZER_PATH, "rb") as f:
    role_vectorizer = pickle.load(f)

with open(DETECTOR_MODEL_PATH, "rb") as f:
    detector_model = pickle.load(f)

with open(DETECTOR_VECTORIZER_PATH, "rb") as f:
    detector_vectorizer = pickle.load(f)


# -----------------------------
# Helpers
# -----------------------------
def extract_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(uploaded_file)

    if suffix == ".docx":
        return extract_docx(uploaded_file)

    raise ValueError("Unsupported file type. Upload PDF or DOCX only.")


def normalize_detector_label(label):
    s = str(label).strip().lower()

    if s in ["resume", "1", "true", "yes"]:
        return "Resume"

    if s in ["not resume", "not_resume", "0", "false", "no"]:
        return "Not Resume"

    return "Not Resume"


def get_resume_probability(cleaned_text):
    vec = detector_vectorizer.transform([cleaned_text])

    if hasattr(detector_model, "predict_proba"):
        probs = detector_model.predict_proba(vec)[0]
        classes = [str(c).strip().lower() for c in detector_model.classes_]

        for cls_name, prob in zip(classes, probs):
            if cls_name == "resume":
                return float(prob)

        return float(max(probs))

    pred = detector_model.predict(vec)[0]
    pred = normalize_detector_label(pred)
    return 1.0 if pred == "Resume" else 0.0


def predict_is_resume(text: str):
    cleaned = clean_text(text)

    if not cleaned.strip():
        return "Not Resume", 0.0

    vec = detector_vectorizer.transform([cleaned])
    pred_label = detector_model.predict(vec)[0]
    pred_label = normalize_detector_label(pred_label)
    resume_prob = get_resume_probability(cleaned)

    if pred_label == "Resume" and resume_prob >= RESUME_THRESHOLD:
        return "Resume", resume_prob

    return "Not Resume", resume_prob


def predict_role(text: str):
    cleaned = clean_text(text)

    if not cleaned.strip():
        return "-", 0.0

    vec = role_vectorizer.transform([cleaned])
    label = role_model.predict(vec)[0]

    if hasattr(role_model, "predict_proba"):
        probs = role_model.predict_proba(vec)[0]
        confidence = float(max(probs))
    else:
        confidence = 0.0

    return str(label), confidence


def build_display_score(role_confidence: float, resume_probability: float) -> float:
    """
    Raw score only. No extra boost.
    Maximum score allowed is 99.00.
    """
    raw_score = ((role_confidence * 100.0) * 0.7) + ((resume_probability * 100.0) * 0.3)
    final_score = raw_score * SCORE_MULTIPLIER
    return round(min(MAX_DISPLAY_SCORE, final_score), 2)


def get_status(document_type: str, score: float):
    if document_type != "Resume":
        return "Not Resume"

    if score > SELECTED_THRESHOLD:
        return "Selected for Predicted Job Role"

    if score >= APPLICABLE_THRESHOLD:
        return "Applicable for Predicted Job Role"

    return "Low Match for Predicted Job Role"


def display_score(value):
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    return f"{value:.2f}"


def prepare_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df[["File Name", "Document Type", "Predicted Job Role", "Score", "Status"]].copy()
    display_df["Score"] = display_df["Score"].apply(display_score)
    return display_df


# -----------------------------
# File uploader + Remove button
# -----------------------------
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Upload one or more files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

with col2:
    st.write("")
    st.write("")
    if st.button("Remove all uploaded files", use_container_width=True):
        st.session_state.uploader_key += 1
        st.rerun()

# -----------------------------
# Main processing
# -----------------------------
if uploaded_files:
    results = []

    for uploaded_file in uploaded_files:
        row = {
            "File Name": uploaded_file.name,
            "Document Type": "Not Resume",
            "Predicted Job Role": "-",
            "Score": None,
            "SortScore": 0.0,
            "Status": "Not Resume",
            "Extracted Text": "",
        }

        try:
            extracted_text = extract_uploaded_file(uploaded_file)
            row["Extracted Text"] = extracted_text

            if not extracted_text or not extracted_text.strip():
                results.append(row)
                continue

            doc_label, resume_prob = predict_is_resume(extracted_text)
            row["Document Type"] = doc_label

            if doc_label != "Resume":
                row["Status"] = "Not Resume"
                results.append(row)
                continue

            role_label, role_conf = predict_role(extracted_text)
            score = build_display_score(role_conf, resume_prob)

            row["Predicted Job Role"] = role_label
            row["Score"] = score
            row["SortScore"] = score
            row["Status"] = get_status("Resume", score)

            results.append(row)

        except Exception as e:
            row["Document Type"] = "Not Resume"
            row["Predicted Job Role"] = "-"
            row["Score"] = None
            row["SortScore"] = 0.0
            row["Status"] = f"Error: {e}"
            results.append(row)

    df = pd.DataFrame(results).sort_values(by="SortScore", ascending=False).reset_index(drop=True)

    st.subheader("Results")
    display_df = prepare_display_dataframe(df)
    st.dataframe(display_df, width="stretch", hide_index=True)

    csv_df = df[["File Name", "Document Type", "Predicted Job Role", "Score", "Status"]].copy()
    csv_df["Score"] = csv_df["Score"].apply(lambda x: "" if pd.isna(x) else round(float(x), 2))
    csv_data = csv_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download results as CSV",
        data=csv_data,
        file_name="screening_results.csv",
        mime="text/csv",
    )

    st.subheader("Preview extracted text")
    for row in results:
        with st.expander(f"{row['File Name']} - Preview"):
            st.write(
                row["Extracted Text"][:5000]
                if row["Extracted Text"]
                else "No text extracted."
            )
else:
    st.info("No files uploaded yet. Use the uploader above to add PDF or DOCX files.")