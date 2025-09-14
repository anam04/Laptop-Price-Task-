import argparse
import pickle
import numpy as np
import pandas as pd

# metrics functions
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def main(args):
    # Load model
    with open(args.model_path, "rb") as f:
        theta = pickle.load(f)
    print(f"Loaded trained model from: {args.model_path}")

    # load data
    df = pd.read_csv(args.data_path)
    y_true = df["Price"].values
    X = df.drop(columns=["Price"]).values.astype(np.float64)

    # add bias
    X = np.c_[np.ones(X.shape[0]), X]

    # predict
    y_pred = X @ theta

    # evaluate
    mse_val = mse(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)

    # save metrics 
    with open(args.metrics_output_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse_val:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse_val:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2_val:.2f}\n")

    print(f"Metrics saved to: {args.metrics_output_path}")

    # save predictions 
    np.savetxt(args.predictions_output_path, y_pred, fmt="%.4f")

    print(f"Predictions saved to: {args.predictions_output_path}")
    print("Sample predictions:", y_pred[:10])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/regression_model_final.pkl")
    parser.add_argument("--data_path", default="data/train_data.csv")
    parser.add_argument("--metrics_output_path", default="results/train_metrics.txt")
    parser.add_argument("--predictions_output_path", default="results/train_predictions.csv")
    args = parser.parse_args() 
    main(args) 

    