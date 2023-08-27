# Use an official Python runtime as a parent image
FROM python:3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install PyTorch
# You can choose one of the provided links to install PyTorch
# For example, using the official PyTorch website:
RUN pip3 install torch torchvision torchaudio

# Install OpenCV
RUN pip3 install opencv-python

# Run the command when the container launches
CMD ["python", "shotpredict.py"]