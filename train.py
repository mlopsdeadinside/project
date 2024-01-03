import pandas as pd
import numpy as np
import subprocess
import dvc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from mlflow.models import infer_signature
import mlflow.sklearn, mlflow
from mlflow import MlflowClient
from preprocess_data import proc_data

def load_data():
    # Load the data from the csv file function
    data = pd.read_csv('dummy_sensor_data.csv')
    return data

def start():
    print("Would you like to perform Data Generation? (y/n): ")
    user_input = y
    if user_input == 'y' :
        subprocess.call(['python', 'generate_data.py'])
    elif user_input == 'n':
        print("No Data Generated.")
    else:
        print("Please Enter An appropriate value")
        exit(0)

    #Loading the data
    initial_data = load_data()
    print(initial_data.head())
    
    #Preprocessing the data
    processed_data = proc_data(initial_data)
    print(processed_data.head())
    processed_data.to_csv('preproc_data.csv', index=False)
    
    #Separating the features and target
    X = processed_data.drop(['Timestamp', 'Reading'], axis=1) # features under consideration
    y = processed_data['Reading']  # Target variable
    
    # Train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("MLFlow Start:")
    # Set the artifact_path to location where experiment artifacts will be saved
    artifact_path = "model"
    # Set the run name to identify the experiment run
    #run_name = "project" 
    # Connecting to the MLflow server
    client = MlflowClient(tracking_uri="https://dagshub.com/mlopsdeadinside/project.mlflow")
    mlflow.set_tracking_uri("https://dagshub.com/mlopsdeadinside/project.mlflow")
    random_forest_experiment = mlflow.set_experiment("mlops")
    
    mlflow.sklearn.autolog()
    with mlflow.start_run(): 
        print("Model Training:")
        # Defining the parameters for the model 
        params = {"n_estimators": 100, "random_state": 42, "max_depth": 5}
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, None] 
        }
        rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='r2')
        rf.fit(X_train, y_train) # Train the model

        print("Model Evaluation:")
        # Make predictions on the test set
        y_pred = rf.predict(X_test)        
        signature = infer_signature(X_test, y_pred)
        # Evaluate the model using mean squared error
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        score = rf.best_score_
        print(f"Evaluation Errors: MSE[{mse}], MAE[{mae}], R2[{r2}], RMSE[{rmse}]")
        mlflow_metrics = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "rmse": rmse,
            "score": score
        }
        # Log the parameters usend for the model fit
        #mlflow.log_params(params)
        mlflow.log_params(param_grid)
        # Log the error metrics that were calculated during validation
        mlflow.log_metrics(mlflow_metrics)
        # Log an instance of the trained model for later use
        mlflow.sklearn.log_model(sk_model=rf, input_example=X_test, artifact_path=artifact_path)
    
    print("Best Model Deployment:")
    print("Would you like to deploy the model? (y/n)")
    user_input2 = 'y'
    if user_input2 == 'y' :
        print("Model Deploying")
        best_model = rf.best_estimator_
        best_params = rf.best_params_
        # Get the best model from the MLflow experiment
        best_run = client.search_runs(
            experiment_ids=random_forest_experiment.experiment_id,
            order_by=["metrics.training_mse ASC"],
            max_results=1,
        )[0]
        print("Best Run:", best_run.info.run_id)
        # Clear app/best_model folder if it exists
        #subprocess.call(['rm', '-rf', 'app/best_model'])  # Saving the best model,Replacing the existing best
        mlflow.sklearn.save_model(best_model, "app/best_model")  # saving best_run as a pickle file
        mlflow.sklearn.log_model( 
            sk_model=best_model,
            artifact_path="sklearn-model",         # Register the best model with MLflow
            signature= signature,
            registered_model_name="random-forest-best"
        )
        print("Model deployed.")
    else:
        print("Model Not Deployed.")

if __name__ == '__main__':
    start()
