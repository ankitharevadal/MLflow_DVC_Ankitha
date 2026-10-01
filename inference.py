import mlflow
import mlflow.sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Your MLflow Run ID
run_id = "463bf72fa6214911868f685cf5fd0f08"

# Load the logged model
model_uri = f"runs:/{run_id}/random_forest_model"

loaded_model = mlflow.sklearn.load_model(model_uri)

# Generate predictions
predictions = loaded_model.predict(X_test)

print("Predictions:")
print(predictions)