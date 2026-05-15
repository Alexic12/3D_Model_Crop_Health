# 🔬 Especificaciones Técnicas
**3D Model Crop Health - Documentación Técnica Detallada**

---

## 📋 Información del Documento

| Campo | Valor |
|-------|--------|
| **Versión** | 1.0.0 |
| **Fecha** | 2025-10-08 |
| **Audiencia** | Desarrolladores, Ingenieros de Software, QA |
| **Nivel Técnico** | Detalle Técnico Profundo |

---

## 🎯 Módulos del Sistema

### **📡 Módulo API Gateway**
**Ubicación:** `app/api/`  
**Responsabilidad:** Gateway de entrada, gestión de workers, routing

#### **main.py - FastAPI Gateway**
```python
class FastAPIGateway:
    """
    Gateway principal del sistema que maneja:
    - Routing de requests a workers específicos
    - Gestión del ciclo de vida de workers Streamlit
    - Configuración CORS y middleware
    - Health checks y monitoring
    """
    
    # Configuración de puertos
    DESKTOP_PORT: int = 8501    # Worker Streamlit Desktop
    MOBILE_PORT: int = 8502     # Worker Streamlit Mobile
    API_PORT: int = 8000        # Gateway principal
    
    # Endpoints principales
    GET /                       # Landing page HTML
    GET /desktop               # Redirect a worker desktop
    GET /mobile                # Redirect a worker mobile  
    GET /health                # Health check endpoint
    GET /{path:path}           # Catch-all 404 handler
```

#### **process_manager.py - Supervisor de Workers**
```python
@dataclass
class WorkerSpec:
    """Especificación de un worker Streamlit"""
    name: str                  # Identificador único
    script: str               # Path al script Streamlit
    port: int                 # Puerto asignado
    proc: Optional[subprocess.Popen] = None  # Proceso activo
    
class ProcessManager:
    """
    Supervisor robusto para workers Streamlit:
    - Auto-restart en caso de fallos
    - Health monitoring continuo
    - Graceful shutdown
    - Thread-based isolation
    """
    
    def start_all(self) -> None:
        """Inicia todos los workers en threads separados"""
        
    def _launch_worker(self, spec: WorkerSpec) -> None:
        """
        Loop principal de un worker:
        - Ejecuta comando Streamlit
        - Monitorea estado del proceso
        - Reinicia automáticamente en caso de fallo
        - Implementa backoff strategy
        """
        
    def stop_all(self) -> None:
        """Termina todos los workers gracefully"""
```

---

### **⚙️ Módulo de Configuración**
**Ubicación:** `app/config/`  
**Responsabilidad:** Configuración centralizada y logging

#### **config.py - Settings Management**
```python
class Settings(BaseSettings):
    """
    Configuración centralizada usando Pydantic Settings:
    - Type validation automática
    - Environment variables integration
    - Default values management
    """
    
    # Workers Configuration
    DESKTOP_SCRIPT: str = Field(default="desktop_app/ui_desktop.py")
    MOBILE_SCRIPT: str = Field(default="mobile_app/ui_mobile.py") 
    DESKTOP_PORT: int = Field(default=8501)
    MOBILE_PORT: int = Field(default=8502)
    
    # API Server Configuration
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    
    # External Services
    GOOGLE_MAPS_API_KEY: str = Field(default="")
    
    # Operational
    LOG_LEVEL: str = Field(default="INFO")
    
    class Config:
        env_prefix = "CHV_"        # Environment variable prefix
        case_sensitive = False     # Allow case variations
```

---

### **🔬 Módulo de Procesamiento de Datos**
**Ubicación:** `app/data/`  
**Responsabilidad:** Pipeline científico, algoritmos ML, análisis GHG

#### **data_processing.py - Pipeline Principal**

##### **Algoritmo NDVI (Normalized Difference Vegetation Index)**
```python
def rejilla_indice(ruta_imagen: str, ruta_color: str) -> pd.DataFrame:
    """
    Procesamiento de imágenes satelitales GeoTIFF para extracción NDVI:
    
    Input:
    - ruta_imagen: Path a imagen GeoTIFF con banda NDVI
    - ruta_color: Path a imagen colormap de referencia
    
    Output:
    - DataFrame con columnas [UTM-x, UTM-y, longitud, latitud, col, row, NDVI]
    
    Algoritmo:
    1. Lectura de raster usando rasterio
    2. Transformación de coordenadas CRS -> WGS84
    3. Filtrado de valores NoData
    4. Extracción de coordenadas geográficas por pixel
    """
    
    # Especificaciones técnicas
    CRS_INPUT: Any              # Sistema de coordenadas de entrada
    CRS_OUTPUT: str = "EPSG:4326"  # WGS84 para salida
    NODATA_VALUE: float = -9999    # Valor para pixels sin datos
    
    # Transformaciones geográficas
    transformer = Transformer.from_crs(crs, CRS_OUTPUT, always_xy=True)
    
    # Validaciones
    assert os.path.exists(ruta_imagen), "Imagen base requerida"
    assert banda_ndvi.shape[0] > 0, "Imagen debe tener contenido"
```

##### **Algoritmo IDW (Inverse Distance Weighting)**
```python
def _idw_index_core(df: pd.DataFrame, resolution: int = 5, k_neighbors: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Interpolación espacial usando Inverse Distance Weighting:
    
    Parámetros:
    - df: DataFrame con coordenadas [longitud, latitud, NDVI]
    - resolution: Resolución de la grilla de salida (NxN)
    - k_neighbors: Número de vecinos para interpolación
    
    Algoritmo IDW:
    w_i = 1 / d_i^p  donde p=3 (factor de potencia)
    z(x,y) = Σ(w_i * z_i) / Σ(w_i)
    
    Output:
    - dfidw: Matriz interpolada en formato spreadsheet
    - dfidw_2: DataFrame con puntos interpolados [id, long-xm, long-ym, NDVI]
    """
    
    # Especificaciones del algoritmo
    POWER_FACTOR: int = 3          # Factor de potencia IDW
    MIN_DISTANCE: float = 1e-12    # Evitar división por cero
    
    # Validaciones de entrada
    assert len(df) >= k_neighbors, f"Dataset debe tener >= {k_neighbors} puntos"
    assert resolution > 0, "Resolución debe ser positiva"
    
    # Cálculo de distancias euclidianas
    dist = np.sqrt((gx - xm)**2 + (gy - ym)**2)
    
    # Selección de k-vecinos más cercanos
    k_indices = np.argsort(dist)[:k_neighbors]
    
    # Aplicación de pesos IDW
    w = 1.0 / (dist_nn**POWER_FACTOR)
```

##### **Algoritmo HPC (Hidden Markov Chain) para Predicción**
```python
def Emision(i2: int, XDe: np.ndarray, NIT: int, NR: int) -> Tuple[np.ndarray, ...]:
    """
    Modelo predictivo usando Hidden Markov Chain + Neural Networks:
    
    Parámetros:
    - XDe: Serie temporal de datos climáticos/NDVI
    - NIT: Número de iteraciones de entrenamiento
    - NR: Número de rezagos (lags) para el modelo
    
    Arquitectura de Red Neural:
    - Input: NR features (lags)
    - Hidden 1: 100 neuronas + Dropout(0.01) + L2(0.001)
    - Hidden 2: 50 neuronas + Dropout(0.01) + L2(0.001)  
    - Hidden 3: 25 neuronas + Dropout(0.01) + L2(0.001)
    - Output: 1 neurona (predicción)
    
    Preprocessing:
    - Normalización MinMax: x_norm = (x - min) / (max - min)
    - Construcción de lags: X[t] = [x[t], x[t-30], x[t-60], x[t-90]]
    - Ventana de predicción: 360 días
    """
    
    # Requisitos del modelo
    MIN_DATA_POINTS: int = 400      # Mínimo de datos para entrenamiento
    FORECAST_HORIZON: int = 360     # Días a predecir
    LAG_INTERVAL: int = 30          # Intervalo entre lags (días)
    
    # Arquitectura Neural Network
    model = Sequential([
        Dense(100, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.01),
        Dense(50, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.01), 
        Dense(25, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.01),
        Dense(1, activation='relu', kernel_regularizer=l2(0.001))
    ])
    
    # Validaciones
    assert len(XDe) >= MIN_DATA_POINTS, "Datos insuficientes para modelo"
    assert NR > 0, "Número de rezagos debe ser positivo"
```

##### **Análisis de Matrices de Transición**
```python
def MatricesTransicion(XD: pd.DataFrame, XD3: np.ndarray, n_var: int, punto: int) -> Tuple[np.ndarray, ...]:
    """
    Construcción de matrices de transición para Hidden Markov Model:
    
    Proceso:
    1. LDA = 1 - NDVI (transformación)
    2. K-Means clustering (5 clusters) para estados discretos
    3. Matriz de transición: P(estado_t+1 | estado_t)
    4. Matrices de emisión: P(observación | estado)
    
    Variables climáticas (n_var=5):
    - Temperatura máxima
    - Temperatura mínima  
    - Velocidad del viento
    - Humedad relativa
    - Precipitación
    """
    
    N_COMPONENTS: int = 5           # Número de estados del HMM
    N_CLIMATE_VARS: int = 5         # Variables climáticas
    RANDOM_SEED: int = 42           # Reproducibilidad
    
    # Clustering para estados discretos
    km = KMeans(n_clusters=N_COMPONENTS, random_state=RANDOM_SEED)
    
    # Matrices de transición
    MTr = np.zeros((N_COMPONENTS, N_COMPONENTS))  # Transiciones entre estados
    bEm = np.zeros((N_CLIMATE_VARS, N_COMPONENTS, N_COMPONENTS))  # Emisiones
```

---

#### **ghg_capture.py - Análisis de Captura de GEI**

##### **Algoritmo de Clustering de Riesgos**
```python
def ClusteringX(XDi: np.ndarray, NCi: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Caracterización de variables aleatorias para análisis de frecuencia:
    
    Input:
    - XDi: Vector de datos de entrada
    - NCi: Número de clusters deseados
    
    Output:
    - XCx: Centros de clusters ordenados
    - sigmax: Diámetros de clusters (dispersión)
    
    Algoritmo:
    1. K-Means clustering con random_state=42
    2. Ordenamiento de centros ascendente
    3. Cálculo de diámetros: σ_j = Σ|XC_i - XC_j| / 4
    """
    
    DEFAULT_CLUSTERS: int = 5       # Clusters por defecto
    DIAMETER_DIVISOR: int = 4       # Factor para cálculo de diámetro
    RANDOM_SEED: int = 42           # Reproducibilidad
```

##### **Lógica Fuzzy para Evaluación de Riesgos**
```python
def Fuzzyx(Xf2: np.ndarray, sigmaf: np.ndarray, Xs2: np.ndarray, sigmas: np.ndarray, 
           XCf: np.ndarray, XCs: np.ndarray, ME: np.ndarray, MI: np.ndarray, 
           MG: np.ndarray, MP: np.ndarray) -> np.ndarray:
    """
    Evaluación de riesgos usando lógica fuzzy:
    
    Parámetros:
    - Xf2: Datos de frecuencia
    - Xs2: Datos de severidad  
    - sigmaf/sigmas: Dispersiones de clusters
    - XCf/XCs: Centros de clusters
    - ME, MI, MG, MP: Matrices de eventos, impacto, gestión, pérdidas
    
    Algoritmo Fuzzy:
    1. Cálculo de pertenencia gaussiana: μ(x) = exp(-0.5 * ((x-c)/σ)²)
    2. Determinación de cluster más probable: argmax(μ)
    3. Cálculo de LDA: (ME × MP × MI) / MG
    
    LDA (Loss Distribution Aggregate) representa la distribución 
    agregada de pérdidas considerando gestión de riesgos.
    """
    
    # Función de pertenencia gaussiana
    membership_function = lambda x, center, sigma: np.exp(-0.5 * ((x - center) / sigma) ** 2)
```

##### **Matriz de Gestión de Riesgos**
```python
# Niveles de gestión para matriz de impacto
MANAGEMENT_LEVELS = {
    1: "Básico",      # Nivel básico de gestión
    2: "Estándar",    # Gestión estándar
    3: "Avanzado",    # Gestión avanzada
    4: "Especializado", # Gestión especializada
    5: "Experto"      # Nivel experto
}

# Matriz de impacto base (5x5)
IMPACT_MATRIX = np.array([
    [1, 1, 1, 2, 2],  # Frecuencia: Muy Pocos
    [1, 2, 2, 3, 3],  # Frecuencia: Pocos  
    [1, 2, 3, 3, 4],  # Frecuencia: Más o Menos
    [2, 3, 3, 4, 4],  # Frecuencia: Muchos
    [2, 3, 4, 4, 5]   # Frecuencia: Bastantes
])
```

---

#### **hpc_calculator.py - Calculador HPC**
```python
class HCPCalc:
    """
    Calculador de riesgo climático usando HPC (High Performance Computing):
    
    Funcionalidades:
    - Integración con pipeline HPC completo
    - Generación de mapas de riesgo 2D
    - Distribuciones mensuales de riesgo
    - Métricas estadísticas de rendimiento
    """
    
    @staticmethod
    def compute_risk_results_via_hpc(indice: str, anio: str, 
                                   field_name: str = None,
                                   base_folder: str = "./upload_data") -> Tuple[pd.DataFrame, Dict]:
        """
        Pipeline completo de análisis de riesgo:
        
        Input:
        - indice: Tipo de índice (NDVI, RECI, etc.)
        - anio: Año de análisis
        - field_name: Nombre del campo/finca
        - base_folder: Directorio base de datos
        
        Output:
        - df_map: DataFrame con ubicaciones y valores NDVI
        - risk_info: Diccionario con análisis de riesgo por punto
        
        Estructura de risk_info:
        {
            point_id: {
                "monthly_distribution": np.ndarray (1000, 12),  # Simulaciones Monte Carlo
                "df_table": pd.DataFrame  # Métricas mensuales resumidas
            }
        }
        """
```

---

### **🎨 Módulo de Visualización**
**Ubicación:** `app/ui/`  
**Responsabilidad:** Componentes de visualización, UI responsiva

#### **visualization.py - Gráficos Científicos**

##### **Visualización 2D NDVI**
```python
def create_2d_scatter_plot_ndvi_plotly(qgis_df: pd.DataFrame, 
                                      sheet_name: str = "NDVI Sheet",
                                      margin_frac: float = 0.05,
                                      compact_mode: bool = True,
                                      debug: bool = False) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Visualización interactiva 2D usando Plotly:
    
    Features:
    - Colorscale custom NDVI (rojo-naranja-amarillo-verde)
    - Hover tooltips con información detallada
    - Responsive design para móviles
    - Paneo y zoom optimizados
    - Aspectos geográficos preservados
    
    Configuraciones de rendering:
    - Mobile: marker_size=10, height=450px
    - Desktop: marker_size=18, height=auto
    - Color range: NDVI [-1, 1]
    - Background: negro para contraste
    """
    
    # Colorscale personalizada para NDVI
    NDVI_COLORSCALE = [
        [0.0, 'red'],      # NDVI bajo (suelo desnudo)
        [0.33, 'orange'],  # NDVI bajo-medio
        [0.66, 'yellow'],  # NDVI medio-alto  
        [1.0, 'green']     # NDVI alto (vegetación densa)
    ]
    
    # Validaciones de entrada
    required_columns = {"long-xm", "long-ym", "NDVI", "Riesgo"}
    assert all(col in qgis_df.columns for col in required_columns)
```

##### **Google Maps Integration**
```python
def fetch_google_static_map(lat_min: float, lat_max: float, 
                           lon_min: float, lon_max: float,
                           api_key: str, img_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Integración con Google Static Maps API:
    
    Cálculo automático de zoom:
    - Base DPP (Degrees Per Pixel) = 360° / 256px
    - Zoom requerido = log2(base_dpp / desired_dpp)
    - Rango válido: [0, 21]
    
    Parámetros de API:
    - maptype: satellite (imágenes satelitales)
    - format: png (mejor calidad)
    - scale: 1 (resolución estándar)
    """
    
    GOOGLE_MAPS_BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"
    BASE_DPP = 360.0 / 256.0        # Degrees per pixel en zoom 0
    MIN_ZOOM, MAX_ZOOM = 0, 21      # Rango válido de zoom
    DEFAULT_SIZE = (640, 640)       # Tamaño por defecto
```

#### **responsive_components.py - UI Responsiva**
```python
class ResponsiveLayout:
    """
    Sistema de layout responsivo para cross-device compatibility:
    
    Breakpoints:
    - Mobile: < 768px
    - Tablet: 768px - 1024px  
    - Desktop: > 1024px
    
    Grid System:
    - Mobile: columna única
    - Tablet: 2 columnas balanceadas
    - Desktop: 2:1 ratio (contenido:sidebar)
    """
    
    @staticmethod
    def responsive_columns(mobile_cols: List[int], 
                          tablet_cols: List[int], 
                          desktop_cols: List[int]) -> List[st.columns]:
        """
        Creación de columnas adaptativas según viewport.
        
        Detección de dispositivo via viewport_width en session_state.
        """
    
    @staticmethod
    def responsive_container(max_width: str = "1200px"):
        """
        Container responsivo con max-width configurable.
        
        CSS Grid + Flexbox para layout fluido.
        """

class AccessibleComponents:
    """
    Componentes accesibles siguiendo WCAG 2.1+:
    
    Features:
    - ARIA labels apropiados
    - Focus management
    - High contrast support
    - Screen reader compatibility
    - Keyboard navigation
    """
```

---

### **🖥️ Módulo Desktop App**
**Ubicación:** `app/desktop_app/`  
**Responsabilidad:** Interfaz Streamlit para escritorio

#### **ui_desktop.py - Interfaz Principal**
```python
def render_desktop():
    """
    Aplicación principal de escritorio con funcionalidades completas:
    
    Secciones principales:
    1. 📊 Análisis Individual: Upload y análisis de archivos únicos
    2. 🔄 Análisis en Lote: Procesamiento masivo con paralelización
    3. 🌡️ Análisis HPC: Predicción climática con Hidden Markov
    4. 🌱 Análisis GHG: Captura de gases de efecto invernadero
    5. ⚙️ Configuración: Settings y preferencias
    
    Features técnicas:
    - Session state management
    - Progress bars para operaciones largas
    - Error handling robusto
    - Export a Excel profesional
    - Caching de resultados computacionales
    """
    
    # Configuración de página
    st.set_page_config(
        page_title="3D Model Crop Health",
        page_icon="🌾", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inyección de CSS responsivo
    inject_responsive_css()
```

##### **Pipeline de Análisis Individual**
```python
def individual_analysis_section():
    """
    Análisis de archivos Excel individuales:
    
    Flujo:
    1. Upload de archivo Excel
    2. Validación de estructura de datos
    3. Procesamiento NDVI
    4. Generación de visualizaciones
    5. Export de resultados
    
    Formatos soportados:
    - Excel (.xlsx) con hojas múltiples
    - Estructura: [Longitud, Latitud, NDVI, Riesgo]
    - Encoding: UTF-8
    """

def bulk_analysis_section():
    """
    Análisis masivo paralelo:
    
    Configuración:
    - Batch size: 20 archivos por lote
    - Workers: min(2, CPU_count)
    - Timeout: 300s por archivo
    - Memory management: Limpieza inmediata post-procesamiento
    
    Output:
    - INFORME_Espacial: Datos georreferenciados
    - INFORME_IDW: Interpolación espacial
    - INFORME_QGIS: Datos para análisis GIS
    """
```

---

### **📱 Módulo Mobile App**
**Ubicación:** `app/mobile_app/`  
**Responsabilidad:** Interfaz móvil optimizada

#### **ui_mobile.py - Interfaz Móvil**
```python
def render_mobile():
    """
    Aplicación móvil touch-optimized:
    
    Features móviles:
    - Mapas interactivos con Folium
    - Touch gestures para navegación
    - Formularios simplificados
    - Visualizaciones adaptativas
    - Offline capability (Progressive Web App)
    
    UX Patterns:
    - Bottom navigation
    - Card-based layouts
    - Swipe gestures
    - Pull-to-refresh
    - Loading states
    """
    
    # Configuración móvil específica
    st.set_page_config(
        page_title="Crop Health Mobile",
        page_icon="📱",
        layout="wide", 
        initial_sidebar_state="collapsed"  # Sidebar colapsado en móvil
    )
    
    # Meta viewport para mobile
    st.markdown("""
        <meta name="viewport" content="width=device-width, initial-scale=1.0, 
                                     maximum-scale=1.0, user-scalable=no">
    """, unsafe_allow_html=True)
```

##### **Sistema de Gestión Manual de Riesgos**
```python
def manual_risk_management():
    """
    Sistema de edición manual de riesgos optimizado para móvil:
    
    Features:
    - Edición in-place de valores de riesgo
    - Validación en tiempo real
    - Auto-save functionality
    - Sincronización con backend
    - Conflict resolution
    
    Data Structure:
    - 25 puntos por hoja (grid 5x5)
    - Campos: [Latitud, Longitud, NDVI, Riesgo, Nuevo Riesgo]
    - Persistencia: Excel con openpyxl
    """
    
    def _save_riesgo_sheet(sheet_name: str, df_sheet: pd.DataFrame, manual_fn: Path):
        """
        Guardado atómico de hojas de riesgo:
        
        Estrategia:
        1. Validación de archivo existente
        2. Mode selection (create/append)
        3. Atomic write operation
        4. Error recovery
        """
```

---

## 🔗 Interfaces y Contratos

### **Data Contracts**
```python
# Estructura de datos NDVI
NDVIDataPoint = {
    "UTM-x": float,           # Coordenada X en UTM
    "UTM-y": float,           # Coordenada Y en UTM  
    "longitud": float,        # Longitud WGS84
    "latitud": float,         # Latitud WGS84
    "col": int,               # Columna en raster
    "row": int,               # Fila en raster
    "NDVI": float             # Valor NDVI [-1, 1]
}

# Estructura de resultados IDW
IDWResult = {
    "spreadsheet": np.ndarray,    # Matriz interpolada
    "points": List[{             # Puntos interpolados
        "id": int,
        "long-xm": float,
        "long-ym": float, 
        "NDVI": float
    }]
}

# Estructura HPC
HPCAnalysis = {
    "V": np.ndarray,             # Matriz de variables (5x12)
    "ydmes": np.ndarray,         # Datos mensuales (12x5)  
    "results": List[{            # Resultados por punto
        "point_idx": int,
        "VC": List[str],         # Vector de clasificación
        "lon": float,
        "lat": float,
        "XInf": np.ndarray,      # Info matrix (12x12)
        "XLDA": np.ndarray       # Datos simulados (1000x12)
    }]
}
```

### **API Contracts**
```python
# Health Check Response
HealthCheckResponse = {
    "status": Literal["healthy", "degraded", "unhealthy"],
    "timestamp": datetime,
    "workers": {
        "desktop": bool,
        "mobile": bool
    },
    "dependencies": {
        "database": bool,
        "storage": bool,
        "external_apis": bool
    }
}

# Worker Status
WorkerStatus = {
    "name": str,
    "port": int,
    "status": Literal["running", "stopped", "error"],
    "uptime": timedelta,
    "last_restart": datetime,
    "error_count": int
}
```

---

## ⚡ Requisitos No Funcionales

### **Performance**
| Métrica | Requisito | Medición |
|---------|-----------|----------|
| **Response Time** | <2s para consultas interactivas | P95 |
| **Throughput** | 100 concurrent users | Peak load |
| **File Processing** | <30s para archivos <10MB | Average |
| **Memory Usage** | <2GB por worker | Peak utilization |

### **Scalability**
| Aspecto | Límite Actual | Límite Objetivo |
|---------|---------------|-----------------|
| **Concurrent Users** | 50 | 500 |
| **File Size** | 50MB | 500MB |
| **Data Points** | 10K per dataset | 100K per dataset |
| **Parallel Workers** | 2 | 10 |

### **Reliability**
| Métrica | Objetivo | SLA |
|---------|----------|-----|
| **Uptime** | 99.9% | Monthly |
| **Error Rate** | <0.1% | Failed requests |
| **Recovery Time** | <30s | Worker restart |
| **Data Integrity** | 100% | Zero data loss |

### **Security**
| Aspecto | Implementación |
|---------|----------------|
| **Authentication** | JWT tokens |
| **Authorization** | RBAC model |
| **Data Encryption** | TLS 1.3 in transit |
| **Input Validation** | Pydantic schemas |
| **Rate Limiting** | 100 req/min per IP |

---

## 🔍 Validaciones y Constraints

### **Data Validation**
```python
# Validación de archivos NDVI
def validate_ndvi_file(file_path: str) -> bool:
    """
    Validaciones obligatorias:
    - Formato GeoTIFF válido
    - Proyección geográfica definida
    - Valores NDVI en rango [-1, 1]
    - Tamaño mínimo 100x100 pixels
    - Metadatos geoespaciales presentes
    """

# Validación de datos Excel
def validate_excel_structure(df: pd.DataFrame) -> bool:
    """
    Columnas requeridas: [Longitud, Latitud, NDVI]
    Tipos de datos: float64 para coordenadas y NDVI
    Rangos válidos: 
    - Longitud: [-180, 180]
    - Latitud: [-90, 90]  
    - NDVI: [-1, 1]
    """
```

### **Business Rules**
```python
# Reglas de negocio para análisis HPC
HPC_BUSINESS_RULES = {
    "min_data_points": 400,        # Mínimo para modelo confiable
    "max_forecast_horizon": 365,   # Máximo días a predecir
    "min_confidence_level": 0.85,  # Confianza mínima del modelo
    "max_processing_time": 300     # Timeout en segundos
}

# Reglas para análisis GHG
GHG_BUSINESS_RULES = {
    "min_scenarios": 3,            # Mínimo escenarios de gestión
    "max_scenarios": 10,           # Máximo escenarios
    "co2_conversion_factor": 250,  # USD por tonelada CO2
    "risk_matrix_size": (5, 5)     # Dimensiones frecuencia x severidad
}
```

---

## 📞 Soporte Técnico

| Aspecto | Contacto | Responsabilidad |
|---------|----------|-----------------|
| **Algoritmos Científicos** | data-science@eafit.edu.co | NDVI, IDW, HPC |
| **Performance Issues** | performance@eafit.edu.co | Optimización |
| **API Integration** | api-support@eafit.edu.co | Endpoints y contratos |
| **UI/UX Issues** | frontend@eafit.edu.co | Interfaz y usabilidad |

---

*Documento técnico de referencia. Actualizado automáticamente con cada release.*

**Próxima Revisión:** 2025-11-08  
**Aprobado por:** Senior Developer | Solution Architect
