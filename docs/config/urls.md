# 🌐 Configuración de URLs - AgroProRisk

## URLs de Producción

### **Dominios Principales**
```
www.agroprorisk.com         # Página principal y landing
desktop.agroprorisk.com     # Interfaz de escritorio
mobile.agroprorisk.com      # Interfaz móvil optimizada
```

### **Puertos y Servicios**
```
Puerto 8000: Gateway principal (FastAPI)
Puerto 8501: Worker Desktop (Streamlit)
Puerto 8502: Worker Mobile (Streamlit)
```

### **Arquitectura de URLs**

#### **Gateway Principal (www.agroprorisk.com:8000)**
- `/` - Landing page informativa
- `/health` - Health check del sistema
- `/desktop` - Redirect a desktop.agroprorisk.com
- `/mobile` - Redirect a mobile.agroprorisk.com

#### **Desktop Worker (desktop.agroprorisk.com:8501)**
- Aplicación completa de escritorio
- Análisis completo de datos NDVI
- Herramientas avanzadas de visualización
- Interfaz para expertos y técnicos

#### **Mobile Worker (mobile.agroprorisk.com:8502)**
- Aplicación optimizada para móviles
- Vista simplificada de datos
- Interfaz táctil responsiva
- Funcionalidades básicas de consulta

---

## Configuración de Desarrollo

### **URLs Locales**
```
http://localhost:8000       # Gateway local
http://localhost:8501       # Desktop worker local
http://localhost:8502       # Mobile worker local
```

### **Variables de Entorno**
```bash
# Producción
AGROPRORISK_MAIN_URL=https://www.agroprorisk.com
AGROPRORISK_DESKTOP_URL=https://desktop.agroprorisk.com
AGROPRORISK_MOBILE_URL=https://mobile.agroprorisk.com

# Desarrollo
AGROPRORISK_MAIN_URL=http://localhost:8000
AGROPRORISK_DESKTOP_URL=http://localhost:8501
AGROPRORISK_MOBILE_URL=http://localhost:8502
```

---

**Desarrollado por:** Alejandro Puerta para Universidad EAFIT  
**Proyecto:** AgroProRisk Platform
