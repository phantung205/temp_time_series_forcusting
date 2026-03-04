import pandas as pd
from sklearn.model_selection import train_test_split
import os
from src import config

def load_data():
    return pd.read_csv(config.raw_data_path)

def create_ts_data(data,window_size = 5, target_size = config.ts):
    for i in range(1,window_size):
        data[f"meantemp_{i}"] = data["meantemp"].shift(-i)

    for j in range(target_size):
        data[f"target_{j+1}"] = data["meantemp"].shift(-(window_size+j))
    return data


def clean_raw_data(df,is_train=True):
    # create data time series forcasting
    df = create_ts_data(df)

    # Convert data from data date time to time type
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # Separate the necessary columns
    df["month"] = df["date"].dt.month

    # delete comlumn necessary
    df = df.drop(columns=['date'])

    # delete empty columns
    df = df.dropna()

    # retain the necessary columns
    if is_train:
        requried_cols = (
            config.numerical_col+
            config.targets
        )
        df = df[requried_cols]
    else:
        df = df[config.numerical_col]
    return df

def preprocess_and_split(test_size=None,random_state=None):
    processed_dir = config.processed_data_dir

    if test_size is None:
        test_size = config.test_size
    if random_state is None:
        random_state = config.random_state

    # load data
    df = load_data()

    # clear data
    df = clean_raw_data(df,True)

    # split target , sample
    x = df.drop(config.targets,axis=1)
    y = df[config.targets]

    # train, test split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,random_state=random_state,shuffle=False)

    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)
    x_train.to_csv(os.path.join(processed_dir, "x_train.csv"), index=False)
    x_test.to_csv(os.path.join(processed_dir, "x_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = preprocess_and_split()
    print(x_train.head(2))
    print(x_test.head(2))
    print(y_train.head(2))
    print(y_test.head(2))