import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from datetime import datetime
from sklearn.metrics import  accuracy_score, precision_score, recall_score


mlflow.set_experiment("CustomerChurn")

df = pd.read_csv("data/telecom_churn.csv")

X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Start the MLflow run manually
current_timestamp = datetime.now()
with mlflow.start_run(run_name='Regression in Unbalanced data' + str(current_timestamp)) as run:
    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    #set description
    mlflow.set_tag("description", "Churn Classification in unbalanced dataset")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # Log the metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
   

    # Save the model using joblib
    joblib.dump(model, 'model/prediction1.joblib')


    # Print the classification report
    print("Unbalanced accuracy = " +  str(accuracy))
    print("Unbalanced precision = " +  str(precision))
    print("Unbalanced recall = " +  str(recall))


# Start the MLflow run manually
current_timestamp = datetime.now()
with mlflow.start_run(run_name='Regression in balanced data'+ str(current_timestamp)) as run:
    # Perform random oversampling
    oversampler = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    # Train the model
    model1 = LogisticRegression()
    model1.fit(X_train_resampled, y_train_resampled)

    #set description
    mlflow.set_tag("description", "Churn Classification in balanced dataset")

    # Make predictions
    y_pred = model1.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # Log the metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Save the model using joblib
    joblib.dump(model1, 'model/prediction2.joblib')

    # Print the  report
    print("balanced accuracy = " +  str(accuracy))
    print("balanced precision = " +  str(precision))
    print("balanced recall = " +  str(recall))
