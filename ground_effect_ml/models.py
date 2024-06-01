from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


def get_linear_model():
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    return model


def get_nn_model(random_state=0):
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=random_state,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", mlp),
        ]
    )
    return model
