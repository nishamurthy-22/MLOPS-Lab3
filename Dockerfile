# Use a slim Python image to keep the final image small
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies first so this layer is cached
# and only rebuilds when requirements.txt changes
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script and dataset into the container
COPY src/ .
COPY data/ ./data/

# Run the training script when the container launches
CMD ["python", "main.py"]