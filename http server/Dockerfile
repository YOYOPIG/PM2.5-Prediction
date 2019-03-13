# Use an official Python runtime as a parent image
FROM python:3.5

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app when the container launches
CMD ["python3", "-u", "-m", "http_server.http_server"]