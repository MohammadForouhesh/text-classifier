import pandas as pd
import sklearn
from tqdm import tqdm
from xgb_clf import XgbClf

tqdm.pandas()


def inference_pipeline(model_path: str, input_text: str):
    xgb = XgbClf(text_array=None, load_path=model_path)
    # return xgb.inference(input_text)
    return xgb.inference_proba(input_text)


def main(dataframe: pd.DataFrame, save_path: str):
    dataframe.replace('', float('NaN')).dropna(inplace=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataframe.text, dataframe.label, stratify=dataframe.label)
    xgb = XgbClf(text_array=dataframe.text)
    xgb.fit(X_train, y_train)
    xgb.predict(X_test, y_test)
    xgb.save_model(save_path)


if __name__ == '__main__':
    df = pd.read_excel('tweet-zare-relabeled.xlsx').sample(388)
    df['label'] = df['polarity-f'].apply(lambda item: int(item == 'positive'))
    main(df, 'crypto')
    # print(df.text[3])
    # xgb = XgbClf(text_array=None, labels=None, load_path='politics')
    # preds = df.text.progress_apply(lambda item: xgb.inference_proba(item))
    # print(classification_report(df.label, preds))