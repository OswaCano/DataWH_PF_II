from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "KNNRegressor": KNeighborsRegressor(n_neighbors=5)
}

CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "KNNClassifier": KNeighborsClassifier(n_neighbors=5)
}
