# 🏗️ Arquitectura del Sistema
**3D Model Crop Health - Documentación Arquitectónica**

---

## 📋 Información del Documento

| Campo | Valor |
|-------|--------|
| **Versión** | 1.0.0 |
| **Fecha** | 2025-10-08 |
| **Audiencia** | Arquitectos, Tech Leads, Desarrolladores Senior |
| **Nivel Técnico** | Arquitectural Avanzado |

---

## 🎯 Visión Arquitectónica

### **Principios Fundamentales**
- **Microservicios Modulares:** Separación clara de responsabilidades
- **Cloud-Native:** Diseñado para AWS con auto-scaling
- **API-First:** Todas las funcionalidades expuestas vía APIs
- **Mobile-First:** UX optimizada para dispositivos móviles
- **Scientific Accuracy:** Algoritmos validados científicamente

### **Estilo Arquitectónico**
**Patrón Principal:** **Gateway + Workers** con **Event-Driven Components**
- FastAPI como API Gateway
- Streamlit Workers para interfaces especializadas
- Process Manager para supervisión automática
- Pipeline científico paralelo para procesamiento de datos

---

## 📐 Diagramas C4 Model

### **Nivel 1: Contexto del Sistema**

```mermaid
graph TB
    subgraph "Usuarios"
        U1[👨‍🌾 Agricultores]
        U2[🔬 Científicos]
        U3[📊 Analistas]
        U4[📱 Móviles]
    end
    
    subgraph "Sistema Principal"
        SYS[🌾 3D Model Crop Health<br/>Sistema de Análisis de Cultivos]
    end
    
    subgraph "Sistemas Externos"
        SAT[🛰️ Datos Satelitales<br/>Google Earth/Sentinel]
        MAPS[🗺️ Google Maps API]
        CLOUD[☁️ AWS Cloud Services]
        DB[🗄️ Storage Systems]
    end
    
    U1 --> SYS
    U2 --> SYS
    U3 --> SYS
    U4 --> SYS
    
    SYS --> SAT
    SYS --> MAPS
    SYS --> CLOUD
    SYS --> DB
    
    style SYS fill:#e8f5e8,stroke:#4caf50,stroke-width:3px
    style SAT fill:#e3f2fd,stroke:#2196f3
    style MAPS fill:#fff3e0,stroke:#ff9800
    style CLOUD fill:#f3e5f5,stroke:#9c27b0
```

### **Nivel 2: Contenedores**

```mermaid
graph TB
    subgraph "Cliente"
        WEB[💻 Web Browser]
        MOB[📱 Mobile Browser]
    end
    
    subgraph "API Gateway Layer"
        GW[🌐 FastAPI Gateway<br/>Puerto 8000]
        PM[⚙️ Process Manager<br/>Supervisor de Workers]
    end
    
    subgraph "Application Layer"
        DT[🖥️ Desktop App<br/>Streamlit 8501]
        MB[📱 Mobile App<br/>Streamlit 8502]
    end
    
    subgraph "Processing Layer"
        DP[🔄 Data Pipeline<br/>NDVI + IDW + HPC]
        ML[🧠 ML Engine<br/>TensorFlow + Keras]
        GHG[🌱 GHG Analyzer<br/>Fuzzy Logic]
    end
    
    subgraph "Data Layer"
        EXCEL[📊 Excel Storage]
        ASSETS[📁 File Assets]
        CONFIG[⚙️ Configuration]
    end
    
    WEB --> GW
    MOB --> GW
    GW --> DT
    GW --> MB
    PM --> DT
    PM --> MB
    
    DT --> DP
    MB --> DP
    DP --> ML
    DP --> GHG
    
    DP --> EXCEL
    DP --> ASSETS
    GW --> CONFIG
    
    style GW fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    style PM fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    style DP fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    style ML fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
```

### **Nivel 3: Componentes (API Gateway)**

```mermaid
graph TB
    subgraph "FastAPI Gateway"
        MAIN[📡 main.py<br/>- Landing Page<br/>- Route Handlers<br/>- CORS Config]
        LIFESPAN[🔄 Lifespan Manager<br/>- Startup/Shutdown<br/>- Worker Supervision]
        ROUTES[🛣️ Route Controllers<br/>- /desktop redirect<br/>- /mobile redirect<br/>- Health checks]
    end
    
    subgraph "Process Manager"
        MANAGER[👔 ProcessManager<br/>- Worker Lifecycle<br/>- Auto-restart<br/>- Health Monitoring]
        SPECS[📋 WorkerSpec<br/>- Desktop: 8501<br/>- Mobile: 8502<br/>- Config Management]
        THREADS[🧵 Thread Pool<br/>- Parallel Workers<br/>- Error Recovery<br/>- Graceful Shutdown]
    end
    
    subgraph "Configuration"
        SETTINGS[⚙️ Settings<br/>- Ports Config<br/>- API Keys<br/>- Environment Vars]
        LOGGING[📝 Logging<br/>- Structured Logs<br/>- Error Tracking<br/>- Performance Metrics]
    end
    
    MAIN --> LIFESPAN
    MAIN --> ROUTES
    LIFESPAN --> MANAGER
    MANAGER --> SPECS
    MANAGER --> THREADS
    SETTINGS --> MAIN
    LOGGING --> MAIN
    
    style MAIN fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    style MANAGER fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    style SETTINGS fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
```

---

## 🔧 Stack Tecnológico

### **Backend Framework**
```python
# Tecnologías Core
FastAPI           # API Gateway con alta performance
Pydantic         # Validación de datos y configuración
Uvicorn          # ASGI server optimizado
```

### **Frontend Framework**
```python
# UI Frameworks
Streamlit        # Interfaces científicas interactivas
Plotly           # Visualizaciones 3D avanzadas
Folium           # Mapas interactivos móviles
mpld3            # Matplotlib interactivo
```

### **Científico/ML**
```python
# Data Science Stack
NumPy            # Computación numérica fundamental
Pandas           # Manipulación de datos estructurados
SciPy            # Algoritmos científicos (IDW, interpolación)
TensorFlow       # Machine Learning y redes neuronales
Keras            # High-level ML API
Scikit-learn     # Clustering y análisis estadístico
```

### **Procesamiento Geoespacial**
```python
# GIS & Remote Sensing
Rasterio         # Procesamiento de imágenes GeoTIFF
PyProj           # Transformaciones de coordenadas
GDAL             # Biblioteca geoespacial fundamental
```

### **Cloud & DevOps**
```yaml
# Infrastructure
AWS ECS          # Container orchestration
Docker           # Containerización
AWS IAM          # Gestión de identidades
AWS S3           # Object storage
CloudFormation   # Infrastructure as Code
```

---

## 🏛️ Patrones Arquitectónicos

### **1. Gateway Pattern**
```python
# app/api/main.py
@app.get("/desktop")
async def serve_desktop():
    return RedirectResponse(url=f"http://{host}:{DESKTOP_PORT}")

@app.get("/mobile") 
async def serve_mobile():
    return RedirectResponse(url=f"http://{host}:{MOBILE_PORT}")
```

**Ventajas:**
- Punto único de entrada
- Load balancing automático
- Monitoring centralizado
- Rate limiting y security

### **2. Supervisor Pattern**
```python
# app/api/process_manager.py
class ProcessManager:
    def _launch_worker(self, spec: WorkerSpec):
        while not self._shutdown.is_set():
            try:
                cmd = [sys.executable, "-m", "streamlit", "run", spec.script]
                spec.proc = subprocess.Popen(cmd)
                spec.proc.wait()
                # Auto-restart on failure
            except Exception as e:
                logger.exception(f"Worker {spec.name} failed")
                time.sleep(5)  # Backoff strategy
```

**Ventajas:**
- Auto-recovery de workers
- Isolation de fallos
- Scaling independiente
- Health monitoring

### **3. Pipeline Pattern**
```python
# app/data/data_processing.py
def bulk_unzip_and_analyze_new_parallel():
    # 1. Data Extraction
    unzip_satellite_data()
    # 2. NDVI Processing  
    ndvi_results = process_ndvi_matrices()
    # 3. IDW Interpolation
    idw_results = apply_idw_interpolation(ndvi_results)
    # 4. Risk Analysis
    risk_results = apply_hpc_analysis(idw_results)
    # 5. GHG Analysis
    ghg_results = analyze_carbon_capture(risk_results)
    return combined_results
```

**Ventajas:**
- Separación de concerns
- Testabilidad individual
- Paralelización
- Error isolation

---

## 🔄 Flujo de Datos

### **Pipeline Principal**

```mermaid
sequenceDiagram
    participant U as 👤 Usuario
    participant G as 🌐 Gateway
    participant W as 🖥️ Worker
    participant P as 🔄 Pipeline
    participant M as 🧠 ML Engine
    participant S as 💾 Storage
    
    U->>G: HTTP Request
    G->>W: Redirect to Worker
    W->>U: UI Interface
    U->>W: Upload Data
    W->>P: Process Request
    
    P->>P: 1. Unzip & Validate
    P->>P: 2. NDVI Analysis
    P->>P: 3. IDW Interpolation
    P->>M: 4. ML Prediction
    M-->>P: Predictions
    P->>P: 5. HPC Analysis
    P->>P: 6. GHG Calculation
    
    P->>S: Store Results
    P->>W: Return Results
    W->>U: Visualizations
```

### **Flujo de Algoritmos Científicos**

```mermaid
flowchart TD
    A[📡 Satellite Data<br/>GeoTIFF Images] --> B[🔍 NDVI Calculation<br/>Vegetation Index]
    B --> C[📊 Spatial Grid<br/>Coordinate System]
    C --> D[🎯 IDW Interpolation<br/>Inverse Distance Weighting]
    D --> E[🧮 Statistical Analysis<br/>Mean, Variance, Clustering]
    E --> F[🔮 HPC Modeling<br/>Hidden Markov Chain]
    F --> G[🌡️ Climate Integration<br/>Temperature, Humidity, Wind]
    G --> H[🧠 ML Prediction<br/>TensorFlow Neural Network]
    H --> I[⚖️ Risk Assessment<br/>Fuzzy Logic Analysis]
    I --> J[🌱 GHG Analysis<br/>Carbon Capture Modeling]
    J --> K[📊 Results Dashboard<br/>Interactive Visualizations]
    
    style A fill:#e3f2fd
    style D fill:#f3e5f5
    style F fill:#e8f5e8
    style H fill:#fff3e0
    style J fill:#fce4ec
```

---

## 🎨 Decisiones Arquitectónicas (ADRs)

### **ADR-001: FastAPI como API Gateway**
**Estado:** Aprobado  
**Fecha:** 2025-10-08  

**Contexto:** Necesidad de un gateway ligero y performante para coordinar workers Streamlit.

**Decisión:** Usar FastAPI como API Gateway principal.

**Justificación:**
- Performance superior vs Flask (3x más rápido)
- Auto-documentación OpenAPI
- Type hints nativo con Pydantic
- Async/await support para I/O intensive operations

**Consecuencias:**
- ✅ Mejor performance y desarrollador UX
- ✅ Documentación automática de APIs
- ⚠️ Curva de aprendizaje para equipo con experiencia Flask

### **ADR-002: Streamlit Workers para UI**
**Estado:** Aprobado  
**Fecha:** 2025-10-08  

**Contexto:** Necesidad de interfaces científicas rápidas para prototipos y visualizaciones.

**Decisión:** Usar Streamlit workers independientes para desktop y mobile.

**Justificación:**
- Desarrollo rápido de UIs científicas
- Componentes nativos para plotting y visualización
- Zero-configuration deployment
- Excelente para dashboards y analytics

**Consecuencias:**
- ✅ Time-to-market acelerado
- ✅ UX científica rica out-of-the-box
- ⚠️ Limitaciones para UX custom complejas

### **ADR-003: Dual Interface Strategy**
**Estado:** Aprobado  
**Fecha:** 2025-10-08  

**Contexto:** Necesidad de soportar tanto usuarios desktop como móviles.

**Decisión:** Desarrollar interfaces separadas optimizadas para cada dispositivo.

**Justificación:**
- Mobile-first approach requiere UX específica
- Desktop permite análisis más profundo
- Optimización independiente de performance
- User journey diferenciado por contexto

**Consecuencias:**
- ✅ UX optimizada por dispositivo
- ✅ Performance específica por plataforma
- ⚠️ Duplicación de código y mantenimiento

---

## 🔧 Configuración y Settings

### **Estructura de Configuración**
```python
# app/config/config.py
class Settings(BaseSettings):
    # Workers Configuration
    DESKTOP_SCRIPT: str = "desktop_app/ui_desktop.py"
    MOBILE_SCRIPT: str = "mobile_app/ui_mobile.py"
    DESKTOP_PORT: int = 8501
    MOBILE_PORT: int = 8502
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # External Services
    GOOGLE_MAPS_API_KEY: str = "AIzaSyB1Vv2XMsTy1AxEowrzOaI5Sn96ffC6HNY"
    
    # Performance
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_prefix = "CHV_"  # Crop Health Visualizer
        case_sensitive = False
```

### **Environment Variables**
```bash
# Development
CHV_LOG_LEVEL=DEBUG
CHV_API_HOST=localhost
CHV_GOOGLE_MAPS_API_KEY=your_dev_key

# Production
CHV_LOG_LEVEL=INFO
CHV_API_HOST=0.0.0.0
CHV_GOOGLE_MAPS_API_KEY=your_prod_key
```

---

## 📊 Métricas y Monitoreo

### **SLAs/SLOs Definidos**
| Métrica | Target | Measurement |
|---------|--------|-------------|
| **Availability** | 99.9% | Monthly uptime |
| **Response Time** | <2s | P95 API calls |
| **Error Rate** | <0.1% | Failed requests |
| **Worker Recovery** | <30s | Process restart time |

### **Health Checks**
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "workers": {
            "desktop": check_worker_health(DESKTOP_PORT),
            "mobile": check_worker_health(MOBILE_PORT)
        },
        "dependencies": {
            "database": check_db_connection(),
            "storage": check_storage_access()
        }
    }
```

---

## 🔮 Evolución Arquitectónica

### **Roadmap Técnico**

**Q4 2025: Optimización**
- Performance tuning del pipeline científico
- Caching layer para resultados computacionales
- Database integration (PostgreSQL + PostGIS)

**Q1 2026: Escalabilidad**
- Kubernetes migration
- Message queues (Redis/RabbitMQ)
- Microservices decomposition

**Q2 2026: Inteligencia**
- Real-time streaming analytics
- Advanced ML models (Computer Vision)
- Edge computing integration

**Q3 2026: Ecosistema**
- API marketplace
- Plugin architecture
- Third-party integrations

---

## 🛡️ Consideraciones de Seguridad

### **Security by Design**
- **API Gateway:** Rate limiting y authentication
- **Process Isolation:** Workers en contenedores separados
- **Data Protection:** Encriptación en tránsito y reposo
- **Access Control:** RBAC implementation ready

### **Compliance Readiness**
- GDPR data protection patterns
- Audit logging comprehensive
- Data retention policies
- Privacy by design principles

---

## 📞 Contactos Arquitectónicos

| Rol | Responsable | Área de Expertise |
|-----|-------------|-------------------|
| **Solution Architect** | architect@eafit.edu.co | Decisiones arquitectónicas |
| **Principal Engineer** | principal@eafit.edu.co | Implementación técnica |
| **DevOps Architect** | devops@eafit.edu.co | Infrastructure & deployment |
| **Security Architect** | security@eafit.edu.co | Security design patterns |

---

*Documento vivo que evoluciona con el sistema. Última actualización: 2025-10-08*

**Próxima Revisión:** 2026-01-08  
**Aprobado por:** Solution Architect | Tech Lead