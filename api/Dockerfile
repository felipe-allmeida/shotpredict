# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6


# Install PyTorch
# You can choose one of the provided links to install PyTorch
# For example, using the official PyTorch website:
RUN pip3 install --no-cache-dir -r ./api/requirements.txt

EXPOSE 5000

# Run the command when the container launches
CMD ["python", "app.py"]