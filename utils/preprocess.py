import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text):
    """
    Clean text for resume detection and role prediction.
    Preserves useful technical keywords and normalizes common patterns.
    """
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()

    if not text:
        return ""

    # Preserve important technical terms before removing punctuation
    replacements = {
        r"\bc\+\+\b": "cplusplus",
        r"\bc#\b": "csharp",
        r"\.net\b": "dotnet",
        r"\bnode\.js\b": "nodejs",
        r"\breact\.js\b": "reactjs",
        r"\bnext\.js\b": "nextjs",
        r"\bvue\.js\b": "vuejs",
        r"\bexpress\.js\b": "expressjs",
        r"\basp\.net\b": "aspdotnet",
        r"\bms sql\b": "mssql",
        r"\bpower bi\b": "powerbi",
        r"\bmachine learning\b": "machinelearning",
        r"\bartificial intelligence\b": "artificialintelligence",
        r"\bdata science\b": "datascience",
        r"\bquality assurance\b": "qualityassurance",
        r"\bmanual testing\b": "manualtesting",
        r"\bautomation testing\b": "automationtesting",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Normalize emails, phone numbers, and URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " LINK ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\+?\d[\d\-\s]{7,}\d", " PHONE ", text)

    # Remove special characters except spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Keep useful words only
    filtered_words = [
        word for word in words
        if word not in ENGLISH_STOP_WORDS and len(word) > 2
    ]

    return " ".join(filtered_words)