import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from crf_pos.pos_tagger.wapiti import WapitiPosTagger

class HandCraftEmbedding():
    def __init__(self, text_series=None):
        self.text_series = None
        self.hashtags_transformation = None
        self.tf_idf_transformation = None
        self.bigram_transformation = None
        self.pos_tagger = WapitiPosTagger()

        if text_series is not None: self.__init(text_series)

    def __init(self, text_series: pd.Series):
        self.text_series = text_series.fillna('')
        self.hashtags_transformation = self.count_hashtags()
        self.tf_idf_transformation = self.tf_idf_transformer()
        self.bigram_transformation = self.count_bigram()

    def count_hashtags(self):
        HASHTAG = r"(\#\w+)"
        top_hashtags = self.text_series.str.extractall(HASHTAG)[0].value_counts().head(100).index

        special_words_transformer = CountVectorizer(encoding='utf-8',
                                                    lowercase=False,
                                                    token_pattern=HASHTAG,
                                                    vocabulary=top_hashtags).fit(self.text_series.ravel())
        return special_words_transformer

    def tf_idf_transformer(self):
        tfidf_transformer = TfidfVectorizer(sublinear_tf=True, min_df=3, norm='l2',
                                            encoding='utf-8', ngram_range=(1, 2)).fit(self.text_series.ravel())
        return tfidf_transformer

    def count_bigram(self):
        bigram_transformer = CountVectorizer(encoding='utf-8', min_df=3, max_features=300,
                                             ngram_range=(1, 2)).fit(self.text_series.ravel())
        return bigram_transformer
    
    def extract_pos(self, text: str, tag: str = 'N') -> str:
        tagged = self.pos_tagger(text)
        return ' '.join([item[0] for item in tagged if item[1] == tag])

    def __getitem__(self, text: str) -> np.ndarray:
        # text = self.extract_pos(text, tag='N')
        hashtag_vector = self.hashtags_transformation.transform([text]).toarray()[0]
        tf_idf_vector  = self.tf_idf_transformation.transform([text]).toarray()[0]
        bigram_vector  = self.bigram_transformation.transform([text]).toarray()[0]

        return np.concatenate([hashtag_vector, tf_idf_vector, bigram_vector])

    def encode(self, text: str) -> np.ndarray:
        return self[text]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f).__dict__)
