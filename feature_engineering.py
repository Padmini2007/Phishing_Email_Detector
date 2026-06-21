"""
feature_engineering.py
-----------------------
Extracts hand-crafted features from raw email text that are known to be
strong signals of phishing, in addition to TF-IDF text features.

Features extracted per email:
    - num_urls            : count of URLs found in the text
    - has_ip_url          : 1 if any URL uses a raw IP address (suspicious)
    - num_suspicious_words: count of urgency / scare words
    - num_exclamations    : count of '!' characters
    - has_money_symbol    : 1 if $ or money-related words appear
    - text_length         : total character length
    - num_uppercase_words : count of ALL-CAPS words (shouting)
    - url_has_at_symbol   : 1 if a URL contains '@' (classic obfuscation trick)
"""

import re
import numpy as np
import pandas as pd

URL_REGEX = re.compile(r"(https?://[^\s]+)", re.IGNORECASE)
IP_REGEX = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

SUSPICIOUS_WORDS = [
    "urgent", "verify", "suspended", "click here", "confirm", "password",
    "update your", "act now", "limited time", "account locked",
    "security alert", "winner", "congratulations", "claim now",
    "bank account", "social security", "credit card", "免费", "free gift",
    "final notice", "unauthorized", "login attempt", "immediately",
    "restricted", "expire", "validate", "refund",
]


def extract_features(text: str) -> dict:
    text_lower = text.lower()
    urls = URL_REGEX.findall(text)

    return {
        "num_urls": len(urls),
        "has_ip_url": int(bool(IP_REGEX.search(text))),
        "num_suspicious_words": sum(text_lower.count(w) for w in SUSPICIOUS_WORDS),
        "num_exclamations": text.count("!"),
        "has_money_symbol": int("$" in text or "money" in text_lower or "refund" in text_lower),
        "text_length": len(text),
        "num_uppercase_words": sum(1 for w in text.split() if w.isupper() and len(w) > 2),
        "url_has_at_symbol": int(any("@" in u for u in urls)),
    }


def build_feature_dataframe(texts: pd.Series) -> pd.DataFrame:
    """Apply extract_features to every email and return a DataFrame."""
    feats = texts.apply(extract_features)
    return pd.DataFrame(list(feats))
