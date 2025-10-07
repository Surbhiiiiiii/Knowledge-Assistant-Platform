import re

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x0c", " ")
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
