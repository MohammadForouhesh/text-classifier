import os
import pickle

import pandas as pd
import re
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
import time

from feature_engineering import HandCraftEmbedding


class XgbClf():
    def __init__(self, text_array: list = None, labels: list = None, load_path: str = None):
        if not isinstance(text_array, pd.Series): text_array = pd.Series(text_array)

        self.xgb = XGBClassifier(n_estimators=300)
        self.emb = HandCraftEmbedding()
        self.scaler = None
        if load_path is not None: self.load_model(load_path)
        else:
            self.emb = HandCraftEmbedding(text_array)

            encoded = list(map(self.emb.encode, tqdm(text_array)))
            self.labels = list(labels)
            self.scaler = self.prep_scaler(encoded)
            self.encoded_input = self.scaler.transform(encoded)

    def prep_scaler(self, encoded):
        scaler = MinMaxScaler()
        scaler.fit(encoded)
        return scaler

    def build(self):
        X_train, X_test, y_train, y_test = train_test_split(self.encoded_input, self.labels, test_size=0.2,
                                                            random_state=42, stratify=self.labels)
        self.xgb.fit(X_train, y_train)
        self.xgb.score(X_test, y_test)
        print(classification_report(y_test, self.xgb.predict(X_test)))
        return self.xgb

    def load_model(self, load_path: str):
        loading_prep = lambda string: f'model_dir/{load_path}/{string}'
        self.xgb.load_model(loading_prep('model.json'))
        self.emb.load(loading_prep('emb.pkl'))
        with open(loading_prep('scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

    def save_model(self, save_path: str):
        os.makedirs(f'model_dir/{save_path}', exist_ok=True)
        saving_prep = lambda string: f'model_dir/{save_path}/{string}'
        self.xgb.save_model(saving_prep('model.json'))
        self.emb.save(saving_prep('emb.pkl'))
        with open(saving_prep('scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f, pickle.HIGHEST_PROTOCOL)

    def inference(self, input_text: str):
        vector = self.scaler.transform(self.emb.encode(input_text).reshape(1, -1))
        return self.xgb.predict(vector)[0]
