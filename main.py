import pandas as pd
import re
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import  RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
import time

from feature_engineering import HandCraftEmbedding
from xgb_clf import XgbClf

tqdm.pandas()


def inference_pipeline(model_path: str, input_text: str):
    xgb = XgbClf(text_array=None, labels=None, load_path=model_path)
    return xgb.inference(input_text)


def main(dataframe: pd.DataFrame, save_path: str):
    # emb_model = HandCraftEmbedding(dataframe['text'])
    # X = list(dataframe['text'].progress_apply(emb_model.encode))
    # y = list(dataframe['label'])
    # scaler = prep_scaler(X)
    # X = scaler.transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    #
    # xgboost_clf = XGBClassifier(n_estimators=300)
    # # xgboost_clf.load_model('exchange_model.json')
    # xgboost_clf.fit(X_train, y_train)
    # xgboost_clf.score(X_test, y_test)
    # # xgboost_clf.predict(X_test[0])
    # print(classification_report(y_test, xgboost_clf.predict(X_test)))
    # xgboost_clf.save_model(save_path+'_model.json')
    xgb = XgbClf(dataframe.text, dataframe.label)
    xgb.build()
    xgb.save_model('jcpoa')
    xgb.load_model('jcpoa')
    print(xgb.inference('llll'))


df = pd.read_excel('jcpoa_sampling.xlsx')
# main(df, 'jcpoa')
print(df.text[0])
inference_pipeline('jcpoa', df.text[0])