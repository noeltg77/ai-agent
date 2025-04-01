FROM python:3.10-slim

WORKDIR /app

# Install Node.js and npm
RUN apt-get update && \
    apt-get install -y curl gnupg git ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create .env file if it doesn't exist (will be overridden by environment variables)
RUN touch .env

# Use a non-root user for better security
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose the API port
EXPOSE 8000

# Run the API using uvicorn directly to the app
CMD ["uvicorn", "Build.API.app:app", "--host", "0.0.0.0", "--port", "8000"]