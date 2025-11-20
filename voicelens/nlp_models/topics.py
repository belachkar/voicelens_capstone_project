import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def generate_topics(df, text_col):
    """
    Takes a DataFrame with a preprocessed text column (already cleaned, lemmatized)
    and returns a Series of human-readable LDA topic labels.
    """

    # ===========================================================
    # 1 — Prepare texts
    # ===========================================================
    texts = df[text_col].astype(str).str.split().tolist()

    # dictionary
    dictionary = Dictionary(texts)

    # remove extremely rare + extremely common words
    dictionary.filter_extremes(no_below=5, no_above=0.6)

    # bag of words corpus
    corpus = [dictionary.doc2bow(text) for text in texts]

    # ===========================================================
    # 2 — Automatic topic selection using coherence
    # ===========================================================
    candidate_topics = [5, 7, 10, 12, 15, 18]

    coherence_values = []
    models = {}

    for k in candidate_topics:
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=8,
            random_state=42
        )

        cm = CoherenceModel(
            model=lda,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v"
        )
        coh = cm.get_coherence()

        coherence_values.append(coh)
        models[k] = lda

    # best topic number
    best_k = candidate_topics[int(np.argmax(coherence_values))]
    lda = models[best_k]

    # ===========================================================
    # 3 — Extract human-readable topic labels
    # ===========================================================
    def extract_topic_label(topic_words):
        """
        Generates a readable label from top topic words.
        """

        words = [w for w, p in topic_words]

        # combine top 3 words into a readable label
        label = " ".join(words[:3])

        # small cleanup
        return label.replace("_", " ").strip()

    topic_labels = {
        topic_id: extract_topic_label(lda.show_topic(topic_id, topn=6))
        for topic_id in range(best_k)
    }

    # ===========================================================
    # 4 — Assign topic to each document
    # ===========================================================
    final_topics = []

    for bow in corpus:
        topic_probs = lda.get_document_topics(bow, minimum_probability=0.0)
        best_topic_id = max(topic_probs, key=lambda x: x[1])[0]
        final_topics.append(topic_labels[best_topic_id])

    return pd.Series(final_topics)
