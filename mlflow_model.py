import mlflow
import mlflow.sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create / select experiment
mlflow.set_experiment("MLflow_Quickstart")

# Load Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Define hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000
}

# Create model
lr = LogisticRegression(**params)

# Start MLflow run
with mlflow.start_run():

    # Train model
    lr.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 1000)

    # Make predictions
    y_pred = lr.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(
        lr,
        "random_forest_model"
    )

    print(f"Model trained with accuracy: {accuracy:.4f}")