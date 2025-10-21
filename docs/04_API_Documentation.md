# 🌐 Documentación de APIs
**3D Model Crop Health - Especificación de APIs**

---

## 📋 Información del Documento

| Campo | Valor |
|-------|--------|
| **Versión** | 1.0.0 |
| **Fecha** | 2025-10-08 |
| **Audiencia** | Desarrolladores Frontend/Backend, Integradores, QA |
| **Nivel Técnico** | Técnico Específico |
| **OpenAPI Version** | 3.0.3 |

---

## 🎯 Visión General de la API

### **Arquitectura API**
- **Patrón:** API Gateway con workers especializados
- **Framework:** FastAPI con documentación automática
- **Autenticación:** JWT Bearer tokens (preparado)
- **Rate Limiting:** 100 requests/minuto por IP
- **Versionado:** Semántico via URL path `/v1/`

### **Base URLs**
```
Desarrollo:  http://localhost:8000/
Producción:  https://www.agroprorisk.com/
Desktop:     https://desktop.agroprorisk.com/
Mobile:      https://mobile.agroprorisk.com/
```

---

## 🔌 Endpoints Principales

### **🏠 Core Gateway Endpoints**

#### **GET /** - Landing Page
```http
GET /
Host: www.agroprorisk.com
Accept: text/html
```

**Descripción:** Página de aterrizaje con interfaz de selección desktop/mobile.

**Response (200 OK):**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Crop Health Gateway</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>🌿 Crop Health Visualizer</h1>
    <a href="/desktop"><button>📊 Desktop Version</button></a>
    <a href="/mobile"><button>📱 Mobile Version</button></a>
</body>
</html>
```

**Headers de Respuesta:**
```http
Content-Type: text/html; charset=utf-8
X-API-Version: 1.0.0
X-Response-Time: <response_time_ms>ms
```

---

#### **GET /desktop** - Redirect a Desktop App
```http
GET /desktop
Host: www.agroprorisk.com
```

**Descripción:** Redirección automática a la aplicación desktop optimizada.

**Response (307 Temporary Redirect):**
```http
HTTP/1.1 307 Temporary Redirect
Location: http://<host_ip>:8501
X-Worker-Type: desktop
X-Worker-Port: 8501
X-Redirect-Reason: desktop_optimization
```

**Códigos de Estado:**
- `307` - Redirect exitoso a worker desktop
- `503` - Worker desktop no disponible
- `500` - Error interno del gateway

---

#### **GET /mobile** - Redirect a Mobile App
```http
GET /mobile
Host: www.agroprorisk.com
```

**Descripción:** Redirección automática a la aplicación móvil touch-optimizada.

**Response (307 Temporary Redirect):**
```http
HTTP/1.1 307 Temporary Redirect
Location: http://<host_ip>:8502
X-Worker-Type: mobile
X-Worker-Port: 8502
X-Redirect-Reason: mobile_optimization
```

**Casos de Uso:**
- Usuario móvil accede desde dispositivo táctil
- Necesidad de interfaz simplificada
- Edición manual de datos en campo

---

### **🔍 Health & Monitoring Endpoints**

#### **GET /health** - Health Check
```http
GET /health
Host: www.agroprorisk.com
Accept: application/json
```

**Descripción:** Endpoint de health check para monitoring y load balancers.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-08T14:30:00Z",
  "version": "1.0.0",
  "uptime": "24h 15m 30s",
  "workers": {
    "desktop": {
      "status": "running",
      "port": 8501,
      "uptime": "24h 15m 28s",
      "health_check_url": "http://localhost:8501/_stcore/health"
    },
    "mobile": {
      "status": "running", 
      "port": 8502,
      "uptime": "24h 15m 25s",
      "health_check_url": "http://localhost:8502/_stcore/health"
    }
  },
  "dependencies": {
    "file_system": {
      "status": "healthy",
      "writable": true,
      "free_space_gb": 45.2
    },
    "google_maps_api": {
      "status": "healthy",
      "last_check": "2025-10-08T14:29:45Z",
      "quota_remaining": 9875
    }
  },
  "metrics": {
    "requests_total": 15420,
    "requests_per_minute": 25.5,
    "error_rate": 0.02,
    "avg_response_time_ms": 145
  }
}
```

**Códigos de Estado:**
- `200` - Sistema completamente saludable
- `503` - Sistema degradado o no disponible

**Health Check Levels:**
```json
{
  "healthy": "Todos los componentes funcionando",
  "degraded": "Algunos workers con problemas",  
  "unhealthy": "Fallos críticos en el sistema"
}
```

---

#### **GET /metrics** - Prometheus Metrics
```http
GET /metrics
Host: www.agroprorisk.com
Accept: text/plain
```

**Descripción:** Métricas en formato Prometheus para monitoring.

**Response (200 OK):**
```prometheus
# HELP crop_health_requests_total Total number of requests
# TYPE crop_health_requests_total counter
crop_health_requests_total{method="GET",endpoint="/desktop"} 8542
crop_health_requests_total{method="GET",endpoint="/mobile"} 4231
crop_health_requests_total{method="GET",endpoint="/health"} 2647

# HELP crop_health_request_duration_seconds Request duration in seconds
# TYPE crop_health_request_duration_seconds histogram
crop_health_request_duration_seconds_bucket{le="0.1"} 12456
crop_health_request_duration_seconds_bucket{le="0.5"} 14532
crop_health_request_duration_seconds_bucket{le="1.0"} 15234
crop_health_request_duration_seconds_bucket{le="+Inf"} 15420

# HELP crop_health_workers_active Number of active workers
# TYPE crop_health_workers_active gauge
crop_health_workers_active{type="desktop"} 1
crop_health_workers_active{type="mobile"} 1

# HELP crop_health_worker_restarts_total Total worker restarts
# TYPE crop_health_worker_restarts_total counter
crop_health_worker_restarts_total{worker="desktop"} 2
crop_health_worker_restarts_total{worker="mobile"} 1
```

---

### **🔧 Management Endpoints**

#### **POST /workers/restart** - Restart Workers
```http
POST /workers/restart
Host: www.agroprorisk.com
Content-Type: application/json
Authorization: Bearer <admin_token>
```

**Request Body:**
```json
{
  "worker_type": "desktop",  // "desktop", "mobile", "all"
  "force": false,            // Force restart even if healthy
  "reason": "maintenance"    // Optional reason for logging
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Worker restart initiated",
  "worker_type": "desktop",
  "restart_id": "restart_2025_10_08_14_30_00",
  "estimated_downtime_seconds": 15
}
```

**Códigos de Estado:**
- `200` - Restart iniciado exitosamente
- `400` - Parámetros inválidos
- `401` - Token de autorización requerido
- `403` - Permisos insuficientes
- `500` - Error en el restart

---

### **🔄 Worker Management**

#### **GET /workers/status** - Workers Status
```http
GET /workers/status
Host: www.agroprorisk.com
Accept: application/json
```

**Response (200 OK):**
```json
{
  "workers": [
    {
      "name": "desktop",
      "type": "streamlit",
      "port": 8501,
      "status": "running",
      "pid": 12345,
      "cpu_percent": 15.2,
      "memory_mb": 245.7,
      "uptime_seconds": 87328,
      "last_restart": "2025-10-07T14:15:30Z",
      "restart_count": 2,
      "script_path": "/app/desktop_app/ui_desktop.py",
      "health_check": {
        "status": "healthy",
        "last_check": "2025-10-08T14:29:55Z",
        "response_time_ms": 45
      }
    },
    {
      "name": "mobile",
      "type": "streamlit", 
      "port": 8502,
      "status": "running",
      "pid": 12347,
      "cpu_percent": 8.1,
      "memory_mb": 189.3,
      "uptime_seconds": 87325,
      "last_restart": "2025-10-07T14:15:33Z",
      "restart_count": 1,
      "script_path": "/app/mobile_app/ui_mobile.py",
      "health_check": {
        "status": "healthy",
        "last_check": "2025-10-08T14:29:52Z", 
        "response_time_ms": 38
      }
    }
  ],
  "summary": {
    "total_workers": 2,
    "running_workers": 2,
    "failed_workers": 0,
    "total_memory_mb": 435.0,
    "average_cpu_percent": 11.65
  }
}
```

---

## 🔐 Autenticación y Autorización

### **Bearer Token Authentication**
```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

**Estructura del JWT:**
```json
{
  "header": {
    "typ": "JWT",
    "alg": "HS256"
  },
  "payload": {
    "sub": "user_12345",
    "exp": 1730815200,
    "iat": 1730728800,
    "role": "admin",
    "permissions": ["read", "write", "restart_workers"],
    "tenant": "universidad_eafit_agroprorisk"
  }
}
```

### **Roles y Permisos**
```yaml
roles:
  viewer:
    permissions:
      - read_health
      - read_metrics
      - access_desktop
      - access_mobile
  
  operator:
    permissions:
      - read_health
      - read_metrics
      - access_desktop
      - access_mobile
      - read_worker_status
  
  admin:
    permissions:
      - "*"  # All permissions
      - restart_workers
      - manage_configuration
      - access_logs
```

---

## 📊 Rate Limiting

### **Límites por Endpoint**
| Endpoint | Límite | Ventana | Scope |
|----------|--------|---------|-------|
| `GET /` | 60 req | 1 minuto | Por IP |
| `GET /desktop` | 30 req | 1 minuto | Por IP |
| `GET /mobile` | 30 req | 1 minuto | Por IP |
| `GET /health` | 120 req | 1 minuto | Por IP |
| `GET /metrics` | 60 req | 1 minuto | Por IP |
| `POST /workers/*` | 10 req | 1 minuto | Por usuario |

### **Response Headers**
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1730728860
X-RateLimit-Window: 60
```

### **Rate Limit Exceeded (429)**
```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded for endpoint",
  "details": {
    "limit": 60,
    "window_seconds": 60,
    "retry_after_seconds": 15
  }
}
```

---

## 🔄 Streamlit Workers API

### **Desktop Worker (Port 8501)**

#### **Capabilities**
- **File Upload Processing:** Excel, GeoTIFF, ZIP files
- **Scientific Analysis:** NDVI, IDW, HPC, GHG algorithms  
- **Advanced Visualizations:** 3D plots, interactive maps
- **Batch Processing:** Parallel analysis of multiple files
- **Export Functionality:** Professional Excel reports

#### **Embedded Endpoints**
```
GET  http://localhost:8501/_stcore/health
GET  http://localhost:8501/_stcore/allowed-message-origins
POST http://localhost:8501/_stcore/upload
```

### **Mobile Worker (Port 8502)**

#### **Capabilities**
- **Touch-Optimized UI:** Mobile-first responsive design
- **Manual Risk Editing:** In-place editing of risk values
- **Interactive Maps:** Folium-based mapping with touch gestures
- **Simplified Workflows:** Streamlined UX for field use
- **Offline Support:** Progressive Web App capabilities

#### **Mobile-Specific Features**
```
- Viewport optimization: width=device-width
- Touch gestures: pan, zoom, tap
- Simplified forms: large touch targets
- Offline caching: ServiceWorker ready
- Push notifications: Ready for implementation
```

---

## 🚨 Error Handling

### **Estructura de Errores**
```json
{
  "error": {
    "code": "WORKER_UNAVAILABLE",
    "message": "Desktop worker is temporarily unavailable",
    "details": {
      "worker_type": "desktop",
      "port": 8501,
      "last_seen": "2025-10-08T14:25:30Z",
      "retry_after_seconds": 30
    },
    "timestamp": "2025-10-08T14:30:00Z",
    "request_id": "req_12345678"
  }
}
```

### **Códigos de Error Comunes**
| Código | Descripción | Resolución |
|--------|-------------|------------|
| `WORKER_UNAVAILABLE` | Worker no responde | Reiniciar worker |
| `WORKER_OVERLOADED` | Worker con alta carga | Retry con backoff |
| `INVALID_ROUTE` | Ruta no encontrada | Verificar URL |
| `RATE_LIMIT_EXCEEDED` | Límite de requests | Esperar ventana |
| `INTERNAL_ERROR` | Error interno | Contactar soporte |

### **HTTP Status Codes**
| Status | Significado | Uso |
|--------|-------------|-----|
| `200` | OK | Request exitoso |
| `307` | Temporary Redirect | Redirect a worker |
| `400` | Bad Request | Parámetros inválidos |
| `401` | Unauthorized | Autenticación requerida |
| `403` | Forbidden | Permisos insuficientes |
| `404` | Not Found | Ruta no existe |
| `429` | Too Many Requests | Rate limit excedido |
| `500` | Internal Server Error | Error del servidor |
| `503` | Service Unavailable | Worker no disponible |

---

## 📝 Logging y Auditoría

### **Request Logging**
```json
{
  "timestamp": "2025-10-08T14:30:00Z",
  "request_id": "req_12345678",
  "method": "GET",
  "path": "/desktop",
  "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
  "ip_address": "192.168.1.100",
  "response_status": 307,
  "response_time_ms": 45,
  "worker_redirected": "desktop",
  "worker_port": 8501
}
```

### **Error Logging**
```json
{
  "timestamp": "2025-10-08T14:30:00Z",
  "level": "ERROR",
  "request_id": "req_12345678",
  "error_code": "WORKER_UNAVAILABLE",
  "error_message": "Desktop worker is not responding",
  "stack_trace": "...",
  "context": {
    "worker_name": "desktop",
    "worker_port": 8501,
    "last_health_check": "2025-10-08T14:25:30Z"
  }
}
```

---

## 🔧 SDK y Ejemplos

### **Python Client Example**
```python
import requests
from typing import Dict, Any

class CropHealthClient:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if api_key:
            self.session.headers['Authorization'] = f'Bearer {api_key}'
    
    def health_check(self) -> Dict[str, Any]:
        """Get system health status"""
        response = self.session.get(f'{self.base_url}/health')
        response.raise_for_status()
        return response.json()
    
    def get_desktop_url(self) -> str:
        """Get desktop application URL"""
        response = self.session.get(f'{self.base_url}/desktop', allow_redirects=False)
        return response.headers['Location']
    
    def get_mobile_url(self) -> str:
        """Get mobile application URL"""
        response = self.session.get(f'{self.base_url}/mobile', allow_redirects=False)
        return response.headers['Location']

# Usage
client = CropHealthClient('http://localhost:8000')
health = client.health_check()
print(f"System status: {health['status']}")
```

### **JavaScript/Fetch Example**
```javascript
class CropHealthAPI {
    constructor(baseUrl, apiKey = null) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.headers = {
            'Content-Type': 'application/json'
        };
        if (apiKey) {
            this.headers['Authorization'] = `Bearer ${apiKey}`;
        }
    }

    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`, {
            headers: this.headers
        });
        return await response.json();
    }

    async getDesktopUrl() {
        const response = await fetch(`${this.baseUrl}/desktop`, {
            method: 'GET',
            redirect: 'manual'
        });
        return response.headers.get('Location');
    }

    async getWorkerStatus() {
        const response = await fetch(`${this.baseUrl}/workers/status`, {
            headers: this.headers
        });
        return await response.json();
    }
}

// Usage
const api = new CropHealthAPI('http://localhost:8000');
api.healthCheck().then(health => {
    console.log('System status:', health.status);
});
```

### **cURL Examples**
```bash
# Health check
curl -X GET "http://localhost:8000/health" \
     -H "Accept: application/json"

# Get desktop URL
curl -X GET "http://localhost:8000/desktop" \
     -H "Accept: application/json" \
     -w "%{redirect_url}"

# Worker status with authentication
curl -X GET "http://localhost:8000/workers/status" \
     -H "Authorization: Bearer your_token_here" \
     -H "Accept: application/json"

# Restart worker (admin only)
curl -X POST "http://localhost:8000/workers/restart" \
     -H "Authorization: Bearer admin_token" \
     -H "Content-Type: application/json" \
     -d '{"worker_type": "desktop", "reason": "maintenance"}'
```

---

## 📋 OpenAPI Specification

### **Complete OpenAPI 3.0 Schema**
```yaml
openapi: 3.0.3
info:
  title: Crop Health Visualizer API
  description: Gateway API for 3D Model Crop Health system
  version: 1.0.0
  contact:
    name: Universidad EAFIT - AgroProRisk
    email: alejandro.puerta@eafit.edu.co
  license:
    name: Proprietary
    
servers:
  - url: http://localhost:8000
    description: Development server
  - url: https://staging.crophealth.eafit.edu.co
    description: Staging server
  - url: https://api.crophealth.eafit.edu.co
    description: Production server

paths:
  /:
    get:
      summary: Landing page
      description: HTML landing page with desktop/mobile options
      responses:
        '200':
          description: Landing page HTML
          content:
            text/html:
              schema:
                type: string

  /desktop:
    get:
      summary: Redirect to desktop application
      description: Redirects to desktop-optimized Streamlit worker
      responses:
        '307':
          description: Redirect to desktop worker
          headers:
            Location:
              description: Desktop worker URL
              schema:
                type: string
                format: uri
        '503':
          description: Desktop worker unavailable

  /mobile:
    get:
      summary: Redirect to mobile application
      description: Redirects to mobile-optimized Streamlit worker
      responses:
        '307':
          description: Redirect to mobile worker
          headers:
            Location:
              description: Mobile worker URL
              schema:
                type: string
                format: uri
        '503':
          description: Mobile worker unavailable

  /health:
    get:
      summary: Health check
      description: System health status for monitoring
      responses:
        '200':
          description: System health information
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthCheck'
        '503':
          description: System unhealthy

components:
  schemas:
    HealthCheck:
      type: object
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
        timestamp:
          type: string
          format: date-time
        workers:
          type: object
          properties:
            desktop:
              $ref: '#/components/schemas/WorkerHealth'
            mobile:
              $ref: '#/components/schemas/WorkerHealth'
              
    WorkerHealth:
      type: object
      properties:
        status:
          type: string
          enum: [running, stopped, error]
        port:
          type: integer
        uptime:
          type: string
```

---

## 📞 Contactos de Soporte

| Tipo | Contacto | Horario |
|------|----------|---------|
| **Desarrollo y Soporte** | alejandro.puerta@eafit.edu.co | 9:00-18:00 COT |
| **Emergencias** | alejandro.puerta@eafit.edu.co | Bajo demanda |
| **Coordinación Académica** | coordinador.sistemas@eafit.edu.co | 8:00-17:00 COT |
| **Soporte Institucional** | soporte.agroprorisk@eafit.edu.co | 9:00-17:00 COT |

---

*Documentación actualizada automáticamente con cada deployment.*

**Próxima Revisión:** 2025-11-08  
**Aprobado por:** Backend Developer | Tech Lead