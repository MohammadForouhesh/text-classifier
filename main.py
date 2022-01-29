import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
from xgb_clf import XgbClf

tqdm.pandas()


def inference_pipeline(model_path: str, input_text: str):
    xgb = XgbClf(text_array=None, labels=None, load_path=model_path)
    # return xgb.inference(input_text)
    return xgb.inference_proba(input_text)


def main(dataframe: pd.DataFrame, save_path: str):
    xgb = XgbClf(text_array=dataframe.text, labels=dataframe.label)
    xgb.build()
    xgb.save_model(save_path)


if __name__ == '__main__':
    df = pd.read_excel('economics.xlsx')
    main(df, 'economics')
    # print(df.text[3])
    # xgb = XgbClf(text_array=None, labels=None, load_path='politics')
    # preds = df.text.progress_apply(lambda item: xgb.inference_proba(item))
    # print(classification_report(df.label, preds))