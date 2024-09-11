# Base image with Python
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose the port the app will run on
EXPOSE 5001

# Set environment variables for AWS (optional if using .env file)
ENV ACCESS_KEY=${ACCESS_KEY}
ENV SECRET_KEY=${SECRET_KEY}
ENV REGION_NAME=${REGION_NAME}

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]
