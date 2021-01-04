import string

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from spacy.lang.en.stop_words import STOP_WORDS as stopwords


# Create a custom transformer to apply SpaCy tokenizer to our data
class SpacyTokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    @staticmethod
    def filter_tokens(tokens):

        tokens = [
            tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_
            for tok in tokens
        ]
        tokens = [
            tok
            for tok in tokens if (tok not in stopwords and tok not in string.punctuation)
        ]

        return " ".join(tokens)

    def transform(self, X, y=None):
        import en_core_web_sm
        nlp = en_core_web_sm.load()
        tokens = X.text.apply(nlp.tokenizer)
        filtered_tokens = tokens.apply(self.filter_tokens)

        if y is None:
            return filtered_tokens

        return filtered_tokens, y


def build_model():
    """This function builds a new model and returns it.

    The model should be implemented as a sklearn Pipeline object.

    Your pipeline needs to have two steps:
    - preprocessor: a Transformer object that can transform a dataset
    - model: a predictive model object that can be trained and generate predictions

    :return: a new instance of your model
    """

    preprocessor = Pipeline(
        [
            ("spacy_tokenizer", SpacyTokenizer()),
            ("tfidf", TfidfVectorizer(stop_words="english")),
        ]
    )

    return Pipeline(
        [("preprocessor", preprocessor), ("model", SGDClassifier(alpha=0.0005))]
    )
