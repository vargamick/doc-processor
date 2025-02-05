FROM chromadb/chroma:latest

# Set environment variables
ENV CHROMA_DB_IMPL=duckdb+parquet
ENV CHROMA_API_IMPL=rest
ENV CHROMA_SERVER_HOST=0.0.0.0
ENV CHROMA_SERVER_HTTP_PORT=8000

# Create persistent volume directory
RUN mkdir -p /chroma/data

# Expose the port
EXPOSE 8000

# Set healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/heartbeat || exit 1
