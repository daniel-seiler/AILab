# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container to /app
WORKDIR /app

RUN pip install --no-cache-dir "numpy<2"
RUN pip install --no-cache-dir fastapi uvicorn \
    pillow jinja2 \
    torch==2.2.2+cpu torchvision==0.17.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir python-multipart
# Make sure the model directory exists
RUN mkdir -p /app/model

# Copy the current directory contents into the container at /app
COPY project/server.py /app/server.py

COPY project/templates/* /app/templates/

COPY ./project/*.pth /app/model

# Expose the port that FastAPI will listen on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
