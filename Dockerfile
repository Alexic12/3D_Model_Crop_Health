# Use a full Python image to ensure compatibility
FROM python:3.10

WORKDIR /app

# System deps for rasterio/geospatial
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY app/ app/
COPY assets/ assets/
COPY upload_data/ upload_data/
COPY .streamlit/ .streamlit/
# COPY .env .   # if you need it

# Expose all three ports for Option B
EXPOSE 8000 8501 8502

# Run the wrapper INSIDE app/ (fixes the path)
CMD ["python", "-u", "app/run_app_uvicorn.py"]