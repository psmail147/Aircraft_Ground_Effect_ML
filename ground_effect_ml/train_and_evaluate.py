from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from data import generate_dataset
from models import get_linear_model, get_nn_model

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Fit the model and compute metrics on train and test sets."""
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = []
    for split_name, y_true, y_pred in [
        ("train", y_train, y_train_pred),
        ("test", y_test, y_test_pred),
    ]:
        for i, target_name in enumerate(["C_L", "C_D"]):
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            metrics.append(
                {
                    "model": name,
                    "split": split_name,
                    "target": target_name,
                    "MAE": mae,
                    "R2": r2,
                }
            )
    return metrics, y_test_pred


def make_scatter_plots(y_test, y_pred_lin, y_pred_nn, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # True vs predicted for C_L
    fig, ax = plt.subplots()
    cl_true = y_test[:, 0]
    cl_pred_lin = y_pred_lin[:, 0]
    cl_pred_nn = y_pred_nn[:, 0]

    min_cl = float(np.min(cl_true))
    max_cl = float(np.max(cl_true))
    line = np.linspace(min_cl, max_cl, 100)

    ax.scatter(cl_true, cl_pred_lin, alpha=0.5, label="Linear regression")
    ax.scatter(cl_true, cl_pred_nn, alpha=0.5, label="Neural network")
    ax.plot(line, line, linestyle="--", label="Ideal")
    ax.set_xlabel("True $C_L$")
    ax.set_ylabel("Predicted $C_L$")
    ax.set_title("True vs predicted lift coefficient")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "true_vs_predicted_CL.png", dpi=300)

    # True vs predicted for C_D
    fig, ax = plt.subplots()
    cd_true = y_test[:, 1]
    cd_pred_lin = y_pred_lin[:, 1]
    cd_pred_nn = y_pred_nn[:, 1]

    min_cd = float(np.min(cd_true))
    max_cd = float(np.max(cd_true))
    line = np.linspace(min_cd, max_cd, 100)

    ax.scatter(cd_true, cd_pred_lin, alpha=0.5, label="Linear regression")
    ax.scatter(cd_true, cd_pred_nn, alpha=0.5, label="Neural network")
    ax.plot(line, line, linestyle="--", label="Ideal")
    ax.set_xlabel("True $C_D$")
    ax.set_ylabel("Predicted $C_D$")
    ax.set_title("True vs predicted drag coefficient")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "true_vs_predicted_CD.png", dpi=300)

def get_better_nn_model(random_state: int = 0):
    """
    Return a more stable neural-network model for multi-output regression.

    Uses MultiOutputRegressor so that C_L and C_D each get their own MLP,
    avoiding the scale imbalance between the targets. Early stopping and a
    modest amount of regularisation help to improve generalisation.
    """
    base_mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-3,
        max_iter=5000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=random_state,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),           # scale input features
            ("regressor", MultiOutputRegressor(base_mlp)),  # one MLP per target
        ]
    )
    return model

def main(random_state: int = 0):
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    figs_dir = project_root / "figs"
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    print("Generating synthetic ground-effect dataset...")
    X, y, df_full = generate_dataset(random_state=random_state)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Save raw dataset for inspection
    df_full.to_csv(results_dir / "ground_effect_dataset.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Create models
    #nn_model = get_nn_model(random_state=random_state) # old model for testing
    lin_model = get_linear_model()
    nn_model = get_better_nn_model(random_state=random_state)

    all_metrics = []

    print("Training linear regression baseline...")
    lin_metrics, y_test_pred_lin = evaluate_model(
        "LinearRegression", lin_model, X_train, y_train, X_test, y_test
    )
    all_metrics.extend(lin_metrics)

    print("Training neural network model...")
    nn_metrics, y_test_pred_nn = evaluate_model(
        "NeuralNetwork", nn_model, X_train, y_train, X_test, y_test
    )
    all_metrics.extend(nn_metrics)

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = results_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")

    # Create plots
    print("Creating comparison plots...")
    make_scatter_plots(y_test, y_test_pred_lin, y_test_pred_nn, figs_dir)

    print("Done. Key outputs:")
    print(f"  Dataset:   {results_dir / 'ground_effect_dataset.csv'}")
    print(f"  Metrics:   {metrics_path}")
    print(f"  Figures:   {figs_dir / 'true_vs_predicted_CL.png'}")
    print(f"             {figs_dir / 'true_vs_predicted_CD.png'}")


if __name__ == "__main__":
    main()
