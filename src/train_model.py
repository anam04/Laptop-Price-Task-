import numpy as np
import pandas as pd
import pickle
import os

# helper functions
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate(y_true, y_pred):
    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": np.sqrt(mse(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

def linear_regression(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def ridge_regression(X, y, alpha=0.01):
    n_features = X.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0  # don’t regularize bias
    return np.linalg.pinv(X.T @ X + alpha * I) @ X.T @ y

def main():
    # load preprocessed data
    data_path = "data/train_data.csv"


    data = pd.read_csv(data_path)

    target_col = "Price"
    X = data.drop(columns=[target_col]).values.astype(np.float64)
    y = data[target_col].values.astype(np.float64)

    # add bias term
    X = np.c_[np.ones(X.shape[0]), X]

    # train/test split 
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Model 1: Linear Regression
    theta1 = linear_regression(X_train, y_train)
    y_pred1 = X_test @ theta1
    metrics1 = evaluate(y_test, y_pred1)

    # Model 2: Polynomial Regression
    X_train_poly = np.c_[X_train, X_train[:, 1:] ** 2]
    X_test_poly = np.c_[X_test, X_test[:, 1:] ** 2]
    theta2 = linear_regression(X_train_poly, y_train)
    y_pred2 = X_test_poly @ theta2
    metrics2 = evaluate(y_test, y_pred2)

    # Model 3: Ridge Regression
    theta3 = ridge_regression(X_train, y_train, alpha=1)
    y_pred3 = X_test @ theta3
    metrics3 = evaluate(y_test, y_pred3)

    # compare & pick best on test set
    models = [
        ("regression_model1.pkl", theta1, metrics1, "Linear Regression"),
        ("regression_model2.pkl", theta2, metrics2, "Polynomial Regression"),
        ("regression_model3.pkl", theta3, metrics3, "Ridge Regression"),
    ]

    print("\nModel Evaluation Results (on test set):")
    for name, _, m, label in models:
        print(f"{label} ({name}): MSE={m['MSE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")

    # best model by R²
    best_model = max(models, key=lambda x: x[2]["R2"])
    final_model_name = "regression_model_final.pkl"
    print(f"\nBest model: {best_model[3]} (saved as {final_model_name})")

    # save models
    os.makedirs("models", exist_ok=True)
    for name, theta, _, _ in models:
        with open(os.path.join("models", name), "wb") as f:
            pickle.dump(theta, f)

    # save final best model
    with open(os.path.join("models", final_model_name), "wb") as f:
        pickle.dump(best_model[1], f)

    print("Training complete.")

if __name__ == "__main__": 
    main()