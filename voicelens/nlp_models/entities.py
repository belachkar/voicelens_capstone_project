import re

import emoji
import pandas as pd
import spacy

# from voicelens.cleaning.preprocessing import apply_cleaning, remove_duplicate_entities

# Charger le modèle spaCy transformer
nlp = spacy.load("en_core_web_trf")

keep_labels = {"PERSON", "ORG", "GPE", "LOC"}  # labels d'entités à garder


def _filter_entities(df, entities_col="entities", keep_labels=keep_labels):
    def filter_fn(ents):
        return [ent for ent in ents if ent[1] in keep_labels]

    df[entities_col] = df[entities_col].apply(filter_fn)
    return df


def _extract_entities(df, text_col="_clean_text"):
    if text_col not in df.columns:
        raise ValueError(f"Colonne '{text_col}' absente du DataFrame.")
    all_entities = []
    for doc in nlp.pipe(df[text_col], batch_size=50, n_process=2):
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        all_entities.append(ents)
    df["entities"] = all_entities
    return df


def _remove_duplicate_entities(df, entities_col="entities"):
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


def run_enr_model(df, text_col="_clean_text", keep_labels=None):
    if keep_labels is None:
        keep_labels = {"PERSON", "ORG", "GPE", "LOC"}

    entities_df = _extract_entities(df, text_col)
    filtered_df = _filter_entities(entities_df, keep_labels=keep_labels)
    filtered_df = _remove_duplicate_entities(filtered_df)
    return filtered_df
