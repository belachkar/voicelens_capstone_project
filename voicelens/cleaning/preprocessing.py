
import pandas as pd
import re
import emoji
import spacy


# Charger le modèle spaCy transformer
nlp = spacy.load("en_core_web_trf")

# Nettoyage complet du texte
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = emoji.replace_emoji(text, "")
    text = text.strip()
    return text

def apply_cleaning(df, text_col="comment"):
    df = df.copy()
    df[text_col] = df[text_col].astype(str).fillna("")
    df["clean_text"] = df[text_col].apply(clean_text)
    df = df[df["clean_text"].str.len() > 3].reset_index(drop=True)
    return df
# Pipeline complet
def spacy_pipeline(df, comment_col="comment"):
    print("🔧 Étape 1 : Nettoyage du texte…")
    df = apply_cleaning(df, comment_col)

    print("🧠 Étape 2 : Extraction des entités (NER)…")
    df = extract_entities(df, "clean_text")

    print("🧹 Étape 3 : Suppression des entités dupliquées…")
    df = remove_duplicate_entities(df, "entities")

    print("⚙️ Étape 4 : Filtrage des entités non significatives…")
    df = filter_entities(df, "entities")

    print("✅ Pipeline terminé !")
    return df


# Suppression des entités dupliquées dans une review
def remove_duplicate_entities(df, entities_col="entities"):
    def unique_entities(ents):
        seen = set()
        unique_ents = []
        for ent in ents:
            if ent not in seen:
                unique_ents.append(ent)
                seen.add(ent)
        return unique_ents

    df[entities_col] = df[entities_col].apply(unique_entities)
    return df

# 4. Clean COMMENT column - REPLACE NEWLINES WITH SPACES
def clean_comment(comment):
    if pd.isna(comment):
        return ""
# 5. Clean RATING column
def clean_rating(rating):
    if pd.isna(rating):
        return None

    try:
        # Handle string ratings
        if isinstance(rating, str):
            # Extract first numeric value
            match = re.search(r'(\d+)', rating)
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

print("Cleaning comment column...")
df['comment'] = df['comment'].apply(clean_comment)

print("Cleaning rating column...")
df['rating'] = df['rating'].apply(clean_rating)

# Remove rows with critical missing data
print(f"\nBefore cleaning: {len(df)} rows")

# Remove duplicates
df_clean = df.drop_duplicates(subset=['name', 'comment', 'date'], keep='first')
print(f"After removing duplicates: {len(df_clean)} rows\n")
# Create rating sentiment with text labels
def get_rating_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df_clean['sentiment_rating'] = df_clean['rating'].apply(get_rating_sentiment)

# Reset index
df_clean = df_clean.reset_index(drop=True)

print(df_clean['sentiment_rating'].value_counts())
# Save the fully cleaned dataset
output_path = "../raw_data/review_for_amazon_fully_cleaned.csv"
df_clean.to_csv(output_path, encoding='utf-8', index=False)

df_clean.head()
