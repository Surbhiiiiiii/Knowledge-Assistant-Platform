# app/services/knowledge.py
import spacy, re, os

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> dict:
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    return entities

def extract_custom_patterns(text: str) -> dict:
    patterns = {}
    m = re.search(r"invoice\s*number[:\-\s]*([A-Za-z0-9\-\/]+)", text, re.IGNORECASE)
    if m:
        patterns["invoice_no"] = m.group(1)
    dates = re.findall(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', text)
    if dates:
        patterns["dates"] = dates
    amounts = re.findall(r"\$\s*[0-9,]+(?:\.\d{1,2})?", text)
    if amounts:
        patterns["amounts"] = amounts
    return patterns
