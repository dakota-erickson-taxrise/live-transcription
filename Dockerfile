# Use a slim Python 3.8 base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy your Python script to the container
COPY main.py .

# Install required libraries
RUN pip install websockets assemblyai anthropic

# Expose the port used by the WebSocket server (default 8765)
EXPOSE 8765

# Run the Python script as the entry point
CMD ["python", "main.py"]