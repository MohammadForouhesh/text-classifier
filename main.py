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
    xgb = XgbClf(dataframe.text, dataframe.label)
    xgb.build()
    xgb.save_model(save_path)


if __name__ == '__main__':
    df = pd.read_excel('jcpoa_sampling.xlsx')
    # main(df, 'jcpoa')
    print(df.text[3])
    print(inference_pipeline('jcpoa', df.text[3]))