# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose port 8080 (Elastic Beanstalk’s default for containerized web apps)
EXPOSE 8080

# Run the Streamlit app with CORS disabled and on port 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]
