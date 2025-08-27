# Crop Health Visualization - Responsive Web Application

## 🌿 Overview

Advanced crop health monitoring and risk assessment platform combining satellite imagery analysis, 3D visualization, and comprehensive risk management for agricultural applications. Built with FastAPI, Streamlit, and modern responsive design.

## ✨ Key Features

### 📊 **3D Visualization & Analysis**
- **NDVI Analysis**: Process satellite imagery with NDVI calculations
- **Interactive 3D Plots**: Rotatable, zoomable surface visualizations
- **Time-Series Animation**: Multi-temporal crop health evolution
- **Bulk Processing**: Automated analysis of ZIP file collections
- **Multiple Output Formats**: Espacial, IDW, and QGIS-compatible exports

### 🎯 **Advanced Risk Management**
- **GHG Capture Analysis**: Complete implementation matching original Jupyter notebook
- **Management Matrices**: Visualization of all management levels (1.5, 2, 3, 4)
- **Fuzzy Logic Implementation**: Proper sampling (NDm=100) and calculations
- **Cost-Benefit Analysis**: Interactive charts showing investment vs returns
- **LDA Distributions**: Multi-scenario risk profile visualizations
- **Real-time Metrics**: Key performance indicators dashboard

### 📱 **Responsive Design**
- **Mobile-First**: Optimized for all device types
- **Touch-Friendly**: Enhanced mobile interactions
- **Accessibility**: WCAG 2.1 AA compliant
- **Cross-Browser**: Chrome 88+, Firefox 85+, Safari 14+, Edge 88+

### 🔬 **High-Performance Computing**
- **Machine Learning**: TensorFlow/Keras models
- **Monte Carlo Simulations**: 1000-iteration risk assessments
- **Hidden Markov Models**: State transition analysis
- **Statistical Analysis**: Skewness, percentiles, operational risk metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional)

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/your_username/3D_Model_Crop_Health.git
cd 3D_Model_Crop_Health
```

2. **Set Up Virtual Environment**
```bash
python -m venv .venv

# Windows
Set-ExecutionPolicy Unrestricted -Scope Process
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run Application**
```bash
python app/run_app_uvicorn.py
```

Access the application:
- **Gateway**: http://localhost:8000
- **Desktop**: http://localhost:8501  
- **Mobile**: http://localhost:8502

### Docker Deployment

```bash
# Build image
docker build -t crop-health-app .

# Run container
docker run -d --name crop-health -p 8000:8000 -p 8501:8501 -p 8502:8502 crop-health-app

# Check status
docker ps
```

## 🏗️ Architecture

### Application Structure
```
app/
├── api/                    # FastAPI gateway
│   ├── main.py            # Main API entry point
│   └── process_manager.py # Streamlit worker management
├── desktop_app/           # Desktop interface
│   └── ui_desktop.py      # Full-featured UI
├── mobile_app/            # Mobile interface
│   └── ui_mobile.py       # Touch-optimized UI
├── ui/                    # Shared UI components
│   ├── responsive_components.py
│   ├── mobile_optimizations.py
│   ├── accessibility.py
│   └── visualization.py
├── data/                  # Data processing
│   ├── data_processing.py # NDVI analysis
│   ├── ghg_capture.py     # Risk management
│   └── hpc_calculator.py  # HPC computations
└── config/                # Configuration
    └── config.py          # Settings management
```

### Technology Stack
- **Backend**: FastAPI, Python 3.10
- **Frontend**: Streamlit (dual instances)
- **Data Processing**: Pandas, NumPy, Rasterio, SciPy
- **Machine Learning**: TensorFlow, Keras, Scikit-learn
- **Visualization**: Plotly, Matplotlib, Folium
- **Geospatial**: GDAL, PyProj
- **Deployment**: Docker, Docker Compose

## 📊 Risk Management Features

### Management Matrix Analysis
- **Baseline Scenario**: Current risk exposure
- **Management Levels**: 1.5x, 2x, 3x, 4x intervention scenarios
- **Interactive Heatmaps**: Visual representation of risk matrices
- **Cost-Benefit Visualization**: Investment vs value captured analysis

### Fuzzy Logic Implementation
```python
# Proper sampling methodology
NDm = 100  # Number of sampling data points
Xfm = pd.DataFrame(Xf).sample(n=NDm, replace=True)
Xs2 = pd.DataFrame(Xs).sample(n=NDm, replace=True)

# Fuzzy membership calculations
VPf = np.exp(-0.5 * ((XCf - Xf2[k]) / sigmaf) ** 2)
VPs = np.exp(-0.5 * ((XCs - Xs2[k]) / sigmas) ** 2)
```

### Loss Distribution Analysis (LDA)
- **Multi-scenario Distributions**: Baseline + 4 management levels
- **Color-coded Visualization**: Red, orange, yellow, blue, green
- **Statistical Metrics**: Mean, 75th percentile, 99.9th percentile
- **Risk Event Calculations**: Normal, severe, and catastrophic events

## 🎨 Responsive Design System

### Breakpoints
- **Mobile**: < 768px (single column, collapsed sidebar)
- **Tablet**: 768px - 1024px (two columns, side navigation)
- **Desktop**: > 1024px (multi-column, fixed sidebar)
- **Large Desktop**: > 1400px (centered, max-width containers)

### CSS Architecture
```css
:root {
    --primary-color: #1f77b4;
    --secondary-color: #ff7f0e;
    --spacing-unit: 1rem;
    --border-radius: 8px;
    --transition-base: 0.15s ease-in-out;
}
```

### Accessibility Features
- **WCAG 2.1 AA Compliance**: Full accessibility support
- **Keyboard Navigation**: Complete keyboard accessibility
- **Screen Reader Support**: ARIA labels and landmarks
- **High Contrast Mode**: `prefers-contrast: high` support
- **Reduced Motion**: `prefers-reduced-motion: reduce` support

## 📈 Performance Optimizations

### Mobile Performance
- **Lazy Loading**: Heavy components load on demand
- **Data Sampling**: Automatic reduction for mobile (max 1000 points)
- **Touch Optimization**: 44px minimum touch targets
- **Optimized Rendering**: Simplified animations and transitions

### Core Web Vitals
- **LCP**: < 2.5s (Largest Contentful Paint)
- **FID**: < 100ms (First Input Delay)
- **CLS**: < 0.1 (Cumulative Layout Shift)

## 🔧 Configuration

### Environment Variables
```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
UVICORN_WORKERS=1

# Application Ports
CHV_DESKTOP_PORT=8501
CHV_MOBILE_PORT=8502

# API Keys
GOOGLE_MAPS_API_KEY=your_api_key_here
```

### File Structure Requirements
```
assets/
├── data/                  # Processed Excel files
│   ├── INFORME_NDVI_Espacial_2024.xlsx
│   ├── INFORME_NDVI_IDW_2024.xlsx
│   └── INFORME_NDVI_QGIS_2024.xlsx
└── images/               # Satellite imagery
    └── NDVI_2024/        # Organized by year
        ├── *.tiff        # Base NDVI files
        └── *ColorMap*.tiff # Color-mapped visualizations

upload_data/              # User uploads
└── NDVI_2024/           # Organized by index and year
    ├── *.zip            # ZIP file collections
    ├── *.tiff           # Extracted TIFF files
    └── Clima_NDVI_2024.xlsx # Climate data
```

## 🧪 Testing & Verification

### Device Testing Matrix
- **Mobile**: iPhone 12/13, Samsung Galaxy S21, Pixel 6
- **Tablet**: iPad Air, Samsung Galaxy Tab, Surface Pro  
- **Desktop**: 1920x1080, 2560x1440, 4K displays
- **Accessibility**: NVDA, JAWS, VoiceOver screen readers

### Performance Benchmarks
- **Mobile Load Time**: < 3 seconds on 3G
- **Desktop Load Time**: < 2 seconds on broadband
- **Lighthouse Score**: 90+ for Performance, Accessibility, Best Practices

### Risk Management Verification
- ✅ **Management Matrices**: All levels (1.5, 2, 3, 4) visualized
- ✅ **Fuzzy Logic**: NDm=100 sampling implemented
- ✅ **Cost-Benefit Analysis**: Interactive charts functional
- ✅ **LDA Distributions**: Multi-scenario visualizations working
- ✅ **Jupyter Notebook Parity**: All original features implemented

## 🚢 Deployment

### Docker Production
```bash
# Build production image
docker build -t crop-health:latest .

# Run with restart policy
docker run -d --restart always \
  --name crop-health-prod \
  -p 8000:8000 -p 8501:8501 -p 8502:8502 \
  crop-health:latest
```

### AWS EC2 Deployment
```bash
# Transfer image
scp -i your-key.pem crop-health.tar ec2-user@your-ec2-ip:/home/ec2-user/

# Load and run on EC2
docker load -i crop-health.tar
docker run -d --restart always --name crop-health -p 8000:8000 crop-health:latest
```

### Health Checks
```bash
# Verify services
curl http://localhost:8000/        # Gateway
curl http://localhost:8501/        # Desktop app
curl http://localhost:8502/        # Mobile app

# Monitor logs
docker logs -f crop-health
```

## 📚 Usage Guide

### 1. Data Upload & Processing
1. Navigate to **3D Visualization** page
2. Upload Excel files or ZIP collections
3. Select vegetation index (NDVI) and year
4. Click **Run Bulk Analysis** for batch processing
5. Download processed files (Espacial, IDW, QGIS formats)

### 2. Risk Management Analysis
1. Go to **Risk Management** page
2. Click **Process GHG Data** to run analysis
3. View **Management Matrices** for all scenarios
4. Analyze **Cost-Benefit** charts for optimal strategy
5. Review **LDA Distributions** for risk profiles
6. Monitor **Risk Metrics Dashboard** for KPIs

### 3. Mobile Risk Assessment
1. Access mobile interface on tablet/phone
2. Select NDVI sheet from dropdown
3. Tap points on interactive map
4. Adjust risk levels (0-6 scale)
5. Download updated risk assessments

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black app/
isort app/

# Type checking
mypy app/
```

### Code Standards
- **Python**: PEP 8 compliance with Black formatting
- **Type Hints**: Full type annotation coverage
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit tests for all modules
- **Accessibility**: WCAG 2.1 AA compliance

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues
- **Port Conflicts**: Ensure ports 8000, 8501, 8502 are available
- **Memory Issues**: Reduce data sampling for large datasets
- **Browser Compatibility**: Use Chrome 88+, Firefox 85+, Safari 14+, Edge 88+

### Getting Help
- **Documentation**: Check this README and inline code comments
- **Issues**: Create GitHub issues for bugs and feature requests
- **Performance**: Monitor Core Web Vitals and optimize accordingly

## 🎯 Roadmap

### Phase 2 Enhancements
- **Real-time Monitoring**: Live satellite data integration
- **Advanced ML Models**: Enhanced prediction algorithms
- **Export Capabilities**: PDF/Excel report generation
- **API Extensions**: RESTful API for third-party integration
- **Progressive Web App**: Offline capabilities and push notifications

---

**Version**: 2.0.0  
**Last Updated**: 2024  
**Maintainer**: F2X Development Team