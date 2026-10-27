# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the model training script into the container
COPY src/ .

# Copy the dataset into the container
COPY data/ ./data/

# Install Scikit-Learn and joblib
RUN pip install -r requirements.txt

# Run the script when the container launches
CMD ["python", "main.py"]
