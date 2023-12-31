# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Define environment variable for MLflow tracking URI
ENV MLFLOW_TRACKING_URI https://dagshub.com/mlopsdeadinside/project.mlflow

# Set the entry point to your script that loads and serves the MLflow model
CMD ["python", "flask.py"]
