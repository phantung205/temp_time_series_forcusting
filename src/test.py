import os
from src import config,preprocessing
import argparse
import joblib
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


def  parse_args():
    p = argparse.ArgumentParser(description="test model")
    p.add_argument("--model_name","-m",type=str,default="XGBoost",help="choise model")

    return  p.parse_args()

def main(args):
    # data
    _,x_test,_,y_test = preprocessing.preprocess_and_split()

    # load model
    model_path = os.path.join(config.model_dir,f"{args.model_name}.joblib")
    if not os.path.isfile(model_path):
        print("You need to train the model to have checkpoints before testing.")
        exit(0)
    model = joblib.load(model_path)

    y_pred = model.predict(x_test)

    # evaluation
    mae_sum = mean_absolute_error(y_test, y_pred)
    mse_sum = mean_squared_error(y_test, y_pred)
    r2_sum = r2_score(y_test, y_pred)

    print("\nEvaluation per target")
    results = []
    for i in range(y_test.shape[1]):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])

        print(f"\nTarget {i + 1}")
        print("MAE:", mae)
        print("MSE:", mse)
        print("R2 :", r2)

        results.append({
            "target": f"target_{i + 1}",
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        })
    print("------------------------")
    print("sum result :")
    print("SUM MAE:", mae_sum)
    print("SUM MSE:", mse_sum)
    print("SUM R2 :", r2_sum)

if __name__ == '__main__':
    args = parse_args()
    main(args)