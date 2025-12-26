# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p models

# Make port 10000 available to the world outside this container (Render uses 10000)
EXPOSE 10000

# Define environment variable
ENV FLASK_APP=simple_app.py

# Run app with gunicorn when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "2", "--timeout", "120", "simple_app:app"]
