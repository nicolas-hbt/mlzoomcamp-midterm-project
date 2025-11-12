# Use the same slim Python 3.12 base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the production requirements file
COPY requirements.txt .
RUN apt-get update
RUN apt-get install -y --no-install-recommends git build-essential
# Install dependencies using standard pip
# This is the most reliable method in Docker.
# --no-cache-dir keeps the image layer small
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get remove -y git build-essential
RUN apt-get autoremove -y

# Copy the application code and the trained model
# We need train.py because predict.py imports its 'preprocess' function
COPY ["predict.py", "predict_test.py", "train.py", "gb_model.pkl", "./"]

# Expose the port the app runs on
EXPOSE 9696

# Define the command to run the application using gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]