# Use an official Python 3.10 image from Docker Hub
FROM python:3.10-slim-bookworm


# Purani line hatao aur yeh dalo:
RUN sed -i 's/deb.debian.org/ftp.us.debian.org/g' /etc/apt/sources.list || true && \
    sed -i 's/security.debian.org/ftp.us.debian.org/g' /etc/apt/sources.list || true

RUN apt-get clean && apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your application code
COPY . /app

# Install the dependencies
RUN pip install -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 5000

# Command to run the FastAPI app
# CMD ["python3", "app.py"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]

