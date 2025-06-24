FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory in the container
WORKDIR /app

# Install system dependencies (for pandas, numpy, openpyxl, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy entire project code into the container
COPY . .

# Expose port for API serving if applicable (adjust if using FastAPI/Flask/Streamlit)
EXPOSE 8080

# Default command to run batch pipeline
# Change to `python app.py` if deploying an API
CMD ["python", "app.py"]