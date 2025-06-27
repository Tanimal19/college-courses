from collections import Counter

import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston


# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, target: str, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Split target and features
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def normalize(X: np.ndarray) -> np.ndarray:
    # Normalize features to [0, 1], using min-max
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    return (X - min_vals) / (max_vals - min_vals)


def standardize(X: np.ndarray) -> np.ndarray:
    # Standardize features to mean=0 and std=1
    mean_vals = np.mean(X, axis=0)
    std_vals = np.std(X, axis=0)
    return (X - mean_vals) / std_vals


def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to integers.
    """
    label_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    return np.array([label_mapping[label] for label in y])


# 3. Models
class LinearModel:
    def __init__(
        self, learning_rate=0.01, iterations=1000, model_type="linear"
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        # You can try different learning rate and iterations
        self.model_type = model_type

        assert model_type in [
            "linear",
            "logistic",
        ], "model_type must be either 'linear' or 'logistic'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.insert(X, 0, 1, axis=1)
        n_classes = len(np.unique(y))
        n_features = X.shape[1]

        if self.model_type == "logistic":
            self.weights = np.zeros((n_features, n_classes))
            y = np.eye(n_classes)[y]
        else:
            self.weights = np.zeros((n_features,))

        for _ in range(self.iterations):
            grad = self._compute_gradients(X, y)
            self.weights -= self.learning_rate * grad

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)

        if self.model_type == "linear":
            return X @ self.weights

        elif self.model_type == "logistic":
            pred = self._softmax(X @ self.weights)
            return np.argmax(pred, axis=1)

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.model_type == "linear":
            return (X.T @ (X @ self.weights - y)) / len(y)

        elif self.model_type == "logistic":
            y_pred = self._softmax(X @ self.weights)
            return (X.T @ (y_pred - y)) / len(y)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)


class DecisionTree:
    def __init__(self, max_depth: int = 5, model_type: str = "classifier"):
        self.max_depth = max_depth
        self.model_type = model_type

        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)
        mask = X[:, feature] <= threshold
        left_X, left_y = X[mask], y[mask]
        right_X, right_y = X[~mask], y[~mask]

        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray):
        if self.model_type == "classifier":
            counts = Counter(y)
            return counts.most_common(1)[0][0]
        else:
            return np.mean(y)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        best_gini = float("inf")
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])
            for i in range(1, len(X)):
                if X[sorted_indices[i - 1], feature] != X[sorted_indices[i], feature]:
                    threshold = (
                        X[sorted_indices[i - 1], feature]
                        + X[sorted_indices[i], feature]
                    ) / 2
                    mask = X[:, feature] <= threshold
                    left_y, right_y = y[mask], y[~mask]

                    if self.model_type == "classifier":
                        gini = self._gini_index(left_y, right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold
                    else:
                        mse = self._mse(left_y, right_y)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature = feature
                            best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        total_len = len(left_y) + len(right_y)
        gini_left = 1 - np.sum([(np.sum(left_y == c) / len(left_y)) ** 2 for c in np.unique(left_y)]) if len(left_y) > 0 else 0
        gini_right = 1 - np.sum([(np.sum(right_y == c) / len(right_y)) ** 2 for c in np.unique(right_y)]) if len(right_y) > 0 else 0
        return (len(left_y) / total_len) * gini_left + (len(right_y) / total_len) * gini_right


    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        total_len = len(left_y) + len(right_y)
        mse_left = np.mean((left_y - np.mean(left_y)) ** 2) if len(left_y) > 0 else 0
        mse_right = np.mean((right_y - np.mean(right_y)) ** 2) if len(right_y) > 0 else 0
        return (len(left_y) / total_len) * mse_left + (len(right_y) / total_len) * mse_right

    def _traverse_tree(self, x: np.ndarray, node: dict):
        if isinstance(node, dict):
            feature, threshold = node["feature"], node["threshold"]
            if x[feature] <= threshold:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            return node


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        model_type: str = "classifier",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_type = model_type
        self.trees = [
            DecisionTree(max_depth=max_depth, model_type=model_type)
            for _ in range(n_estimators)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples = X.shape[0]
        for tree in self.trees:
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[bootstrap_indices], y[bootstrap_indices]
            tree.fit(X_sample, y_sample)

    def predict(self, X: np.ndarray) -> np.ndarray:
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        if self.model_type == "classifier":
            # For classification, return the most frequent class
            return np.array([np.argmax(np.bincount(tree_preds[:, i].astype(int))) for i in range(X.shape[0])])
        else:
            # For regression, return the average prediction
            return np.mean(tree_preds, axis=0)


# 4. Evaluation metrics
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# 5. Main function
import argparse

class bcolors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def print_header(title):
    print(f"{bcolors.OKCYAN}\n== {title} =={bcolors.ENDC}")

def preprocess(dataset, target, pipeline):
    X_train, X_test, y_train, y_test = train_test_split(dataset, target)
    for step in pipeline:
        if step == "normalize":
            X_train = normalize(X_train)
            X_test = normalize(X_test)
        elif step == "standardize":
            X_train = standardize(X_train)
            X_test = standardize(X_test)
        elif step == "encode":
            y_train = encode_labels(y_train)
            y_test = encode_labels(y_test)
    return X_train, X_test, y_train, y_test

def run_classification(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy(y_test, y_pred)

def run_regression(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

def run_preprocess_comparison(iris, boston):
    print_header("Preprocessing Comparison")
    techniques = ["normalize", "standardize"]

    for tech in techniques:
        print(f"{bcolors.BOLD}-- Technique: {tech} --{bcolors.ENDC}")

        X_train, X_test, y_train, y_test = preprocess(iris, "class", [tech, "encode"])
        acc = run_classification(X_train, X_test, y_train, y_test, LinearModel(model_type="logistic"))
        print(f"[Classification] Accuracy: {bcolors.OKGREEN}{acc:.4f}{bcolors.ENDC}")

        X_train, X_test, y_train, y_test = preprocess(boston, "medv", [tech])
        mse = run_regression(X_train, X_test, y_train, y_test, LinearModel(model_type="linear"))
        print(f"[Regression]     MSE:      {bcolors.OKGREEN}{mse:.4f}{bcolors.ENDC}\n")

def run_model_comparison(iris, boston):
    print_header("Model Comparison")
    print(f"{bcolors.BOLD}-- Task: Classification --{bcolors.ENDC}")
    X_train, X_test, y_train, y_test = preprocess(iris, "class", ["normalize", "encode"])
    for model in [LinearModel(model_type="logistic"), DecisionTree(model_type="classifier"), RandomForest(model_type="classifier")]:
        acc = run_classification(X_train, X_test, y_train, y_test, model)
        print(f"{model.__class__.__name__:<20} acc: {bcolors.OKGREEN}{acc:.4f}{bcolors.ENDC}")

    print(f"\n{bcolors.BOLD}-- Task: Regression --{bcolors.ENDC}")
    X_train, X_test, y_train, y_test = preprocess(boston, "medv", ["normalize"])
    for model in [LinearModel(model_type="linear"), DecisionTree(model_type="regressor"), RandomForest(model_type="regressor")]:
        mse = run_regression(X_train, X_test, y_train, y_test, model)
        print(f"{model.__class__.__name__:<20} mse: {bcolors.OKGREEN}{mse:.4f}{bcolors.ENDC}")

def run_linear_hyperparam_comparison(iris, boston):
    print_header("Linear Hyperparameter Comparison")
    configs = [(0.01, 1000), (0.1, 1000), (0.01, 100), (0.1, 100)]

    for lr, iters in configs:
        print(f"{bcolors.BOLD}-- Params: lr={lr}, iter={iters} --{bcolors.ENDC}")

        X_train, X_test, y_train, y_test = preprocess(iris, "class", ["normalize", "encode"])
        model = LinearModel(model_type="logistic", learning_rate=lr, iterations=iters)
        acc = run_classification(X_train, X_test, y_train, y_test, model)
        print(f"[Classification] Accuracy: {bcolors.OKGREEN}{acc:.4f}{bcolors.ENDC}")

        X_train, X_test, y_train, y_test = preprocess(boston, "medv", ["normalize"])
        model = LinearModel(model_type="linear", learning_rate=lr, iterations=iters)
        mse = run_regression(X_train, X_test, y_train, y_test, model)
        print(f"[Regression]     MSE:      {bcolors.OKGREEN}{mse:.4f}{bcolors.ENDC}\n")

def run_tree_hyperparam_comparison(iris, boston):
    print_header("Tree Hyperparameter Comparison")
    configs = [(100, 5), (100, 10), (200, 5), (200, 10)]

    for n_est, max_d in configs:
        print(f"{bcolors.BOLD}-- Params: n_estimators={n_est}, max_depth={max_d} --{bcolors.ENDC}")

        X_train, X_test, y_train, y_test = preprocess(iris, "class", ["normalize", "encode"])
        model = RandomForest(model_type="classifier", n_estimators=n_est, max_depth=max_d)
        acc = run_classification(X_train, X_test, y_train, y_test, model)
        print(f"[Classification] Accuracy: {bcolors.OKGREEN}{acc:.4f}{bcolors.ENDC}")

        X_train, X_test, y_train, y_test = preprocess(boston, "medv", ["normalize"])
        model = RandomForest(model_type="regressor", n_estimators=n_est, max_depth=max_d)
        mse = run_regression(X_train, X_test, y_train, y_test, model)
        print(f"[Regression]     MSE:      {bcolors.OKGREEN}{mse:.4f}{bcolors.ENDC}\n")

def main():
    parser = argparse.ArgumentParser(description="Run model/preprocess/hyperparameter comparison tasks.")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--model", action="store_true", help="Run model comparison task")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing comparison task")
    parser.add_argument("--linear", action="store_true", help="Run linear model hyperparameter tuning")
    parser.add_argument("--tree", action="store_true", help="Run tree model hyperparameter tuning")
    args = parser.parse_args()

    # if no arguments provided, print help message
    if not any(vars(args).values()):
        print("No arguments provided. Use --help to see available options.")
        return

    iris, boston = load_data()

    if args.model or args.all:
        run_model_comparison(iris, boston)
    if args.preprocess or args.all:
        run_preprocess_comparison(iris, boston)
    if args.linear or args.all:
        run_linear_hyperparam_comparison(iris, boston)
    if args.tree or args.all:
        run_tree_hyperparam_comparison(iris, boston)

if __name__ == "__main__":
    main()
