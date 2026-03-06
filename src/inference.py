import os
from src import config,preprocessing
import joblib
import pandas as pd

def load_model(model_name):
    model_path = os.path.join(config.model_dir,f"{model_name}.joblib")
    if not os.path.isfile(model_path):
        raise  FileExistsError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def model_predict_dic(input_dic,model_name):
    # load model
    model = load_model(model_name)

    # convert dic to dataframe
    df = pd.DataFrame([input_dic])

    #clear data
    df = preprocessing.clean_raw_data(df,False)

    #predict
    prediction = model.predict(df)

    return prediction

def model_predict_file(input_file,model_name):
    # load model
    model = load_model(model_name)

    # load data
    if input_file.endswith(".csv"):
        try:
            df = pd.read_csv(input_file)
        except Exception:
            raise ValueError("can not load file this csv ")
    elif input_file.endswith(".xlsx") or input_file.endswith(".xls"):
        try:
            df = pd.read_excel(input_file)
        except Exception:
            raise ValueError("can not load file this exel ")
    else:
        raise ValueError("Only CSV or Excel files are supported")

    #clear data
    df = preprocessing.clean_raw_data(df, False)

    # prediction
    prediction = model.predict(df)

    for i in range(prediction.shape[1]):
        df[f"day_{i + 1}"] = prediction[:, i]

    return df

if __name__ == '__main__':
    sample1 = {
        "month": 3,
        # 5 ngày trước
        "meantemp": 24.5,
        # 4 - 1 ngày trước
        "meantemp_1": 24.8,
        "meantemp_2": 25.0,
        "meantemp_3": 25.3,
        "meantemp_4": 25.6,
        "humidity": 60.2,
        "wind_speed": 7.8,
        "meanpressure": 1012.5
    }
    result_dic = model_predict_dic(sample1,"XGBoost")

    test_file = os.path.join(config.processed_data_dir, "x_test.csv")
    df_result = model_predict_file(test_file, "XGBoost")
    print(df_result.head())