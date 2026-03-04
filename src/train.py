import joblib
from src import preprocessing, config
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import os
from sklearn.multioutput import MultiOutputRegressor


from xgboost import XGBRegressor



def parse_args():
    p = argparse.ArgumentParser(description="train")
    # argument test size and random state
    p.add_argument("--random_state", "-r", type=int, default=config.random_state, help="random state")
    p.add_argument("--test_size", "-t", type=float, default=config.test_size, help="test size")
    # name model
    p.add_argument("--model_name","-m", type=str,default="XGBoost",help="choies model")

    # argument randomForestRegression
    p.add_argument("--n_estimators","-n",type=int,default=300,help="number n_estimators")

    # XGBoost simple params
    p.add_argument("--xgb_n_estimators", type=int, default=500)
    p.add_argument("--xgb_learning_rate", type=float, default=0.05)
    p.add_argument("--xgb_max_depth", type=int, default=6)


    return p.parse_args()

def build_model(args):
    if args.model_name == "LinearRegression":
        model = LinearRegression(n_jobs=-1)
    elif args.model_name == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=args.random_state,
            n_jobs=-1
        )
    elif args.model_name == "XGBoost":
        model = XGBRegressor(
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            random_state=args.random_state,
            n_jobs=-1
        )
    else:
        raise ValueError("Model not supported")
    return MultiOutputRegressor(model)


def main(args):
    #  retrieve data
    x_train, x_test, y_train, y_test = preprocessing.preprocess_and_split(
        test_size=args.test_size,
        random_state=args.random_state
    )

    # preprocessing pipeline
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num_feature", num_transformer, config.numerical_col),
    ])

    # create pipline model
    model = build_model(args)

    # full pipeline
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # train
    pipe.fit(x_train, y_train)

    # predict
    y_pred = pipe.predict(x_test)



    # evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nEvaluation")
    print("MAE:", mae)
    print("MSE:", mse)
    print("R2 :", r2)

    if not os.path.isdir(config.result_report_dir):
        os.makedirs(config.result_report_dir)
    path_result_report = os.path.join(config.result_report_dir,f"train_report_{args.model_name}.txt")
    with open(path_result_report, "w") as f:
        f.write(f"Model: {args.model_name} \n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"R2 : {r2:.4f}\n")

    if not os.path.isdir(config.model_dir):
        os.makedirs(config.model_dir)
    model_name = f"{args.model_name}.joblib"
    model_path = os.path.join(config.model_dir, model_name)
    joblib.dump(pipe, model_path)
    print("save model successfull")


if __name__ == '__main__':
    args = parse_args()
    main(args)