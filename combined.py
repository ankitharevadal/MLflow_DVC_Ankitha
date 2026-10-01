import mlflow
import mlflow.sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Create MLflow experiment
mlflow.set_experiment("MLflow_DVC_Experiment")


# Read DVC metadata
with open("data.txt.dvc", "r") as f:
    dvc_metadata = f.read()

# Extract DVC MD5 hash
dvc_md5 = None

for line in dvc_metadata.splitlines():
    line = line.strip()
    if line.startswith("- md5:"):
        dvc_md5 = line.split(":", 1)[1].strip()
        break


# Load Iris dataset
X, y = datasets.load_iris(return_X_y=True)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Model parameters
solver = "lbfgs"
max_iter = 1000

model = LogisticRegression(
    solver=solver,
    max_iter=max_iter
)


# Start MLflow run
with mlflow.start_run():

    # Log DVC version information
    mlflow.log_param("dvc_data_version", dvc_md5)

    # Log model parameters
    mlflow.log_param("solver", solver)
    mlflow.log_param("max_iter", max_iter)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log accuracy
    mlflow.log_metric("accuracy", accuracy)

    # Log DVC metadata as an artifact
    mlflow.log_text(
        dvc_metadata,
        "dvc_metadata.txt"
    )

    # Log trained model
    mlflow.sklearn.log_model(
        model,
        "model"
    )

    print("DVC + MLflow experiment completed!")
    print("DVC data version:", dvc_md5)
    print(f"Accuracy: {accuracy:.4f}")