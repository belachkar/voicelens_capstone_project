import re

import emoji
import pandas as pd
import spacy

# Charger le modèle spaCy transformer
nlp = spacy.load("en_core_web_trf")


# Nettoyage complet du texte
def _clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = emoji.replace_emoji(text, "")
    text = text.strip()
    return text


# 5. Clean RATING column
def _clean_rating(rating):
    if pd.isna(rating):
        return None

    try:
        # Handle string ratings
        if isinstance(rating, str):
            # Extract first numeric value
            match = re.search(r"(\d+)", rating)
            if match:
                rating = int(match.group(1))
            else:
                return None
        else:
            rating = int(float(rating))

        # Validate range
        if 1 <= rating <= 5:
            return rating
        else:
            return None

    except (ValueError, TypeError):
        return None
    print("\n\nSTARTING COMPREHENSIVE DATA CLEANING")


def apply_cleaning(df, text_col="comment"):
    df = df.copy()
    df[text_col] = df[text_col].astype(str).fillna("")
    df["_clean_text"] = df[text_col].apply(_clean_text)
    df = df[df["_clean_text"].str.len() > 3].reset_index(drop=True)
    return df


def clean(df, comment_col="comment"):
    print("🔧 Étape 1 : Nettoyage du texte…")
    df = apply_cleaning(df, comment_col)

    print("Cleaning rating column...")
    df["rating"] = df["rating"].apply(_clean_rating)

    print("Removing duplicates...")
    df = df.drop_duplicates(subset=["name", "comment", "date"], keep="first")
    return df
