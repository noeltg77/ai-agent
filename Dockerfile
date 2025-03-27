FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Use a non-root user for better security
RUN useradd -m appuser
USER appuser

# Expose the API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "API.app:app", "--host", "0.0.0.0", "--port", "8000"]