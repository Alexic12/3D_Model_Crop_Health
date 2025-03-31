# Use a full Python image to ensure compatibility
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for rasterio and geospatial processing
RUN apt-get update && apt-get install -y \
    libexpat1 \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files
COPY run_app.py .
COPY app/ app/
COPY assets/ assets/
COPY upload_data/ upload_data/
#COPY .env .

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8085
EXPOSE 8501

# Run the application
CMD ["python", "run_app.py"]