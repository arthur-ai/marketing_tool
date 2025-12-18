# Dockerfile for Marketing Project API
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download UDPipe English model (required for SEO keywords engine)
RUN mkdir -p /root/.udpipe/models && \
    wget -q -O /root/.udpipe/models/english-ewt-ud-2.5-191206.udpipe \
    https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/english-ewt-ud-2.5-191206.udpipe

# Copy application code (includes config and prompts now)
COPY src/ ./src/
COPY setup.py ./
COPY pyproject.toml ./

# Install the package
RUN pip install -e .

# Create necessary directories (logs go to Docker logs, not files)
RUN mkdir -p /app/uploads /app/cache

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "marketing_project.server:app", "--host", "0.0.0.0", "--port", "8000"]
