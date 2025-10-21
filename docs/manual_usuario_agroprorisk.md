# 📱 Manual de Usuario - AgroProRisk
## Sistema de Análisis de Riesgo en Cultivos mediante Imágenes Satelitales

---

### 📋 Información del Sistema
- **Desarrollado por:** Universidad EAFIT
- **Autor:** Alejandro Puerta
- **Versión:** 2024
- **Plataforma:** Aplicación Web Desktop

---

## 🎯 ¿Qué es AgroProRisk?

AgroProRisk es una herramienta avanzada que te permite analizar la salud de tus cultivos y predecir riesgos utilizando imágenes satelitales. El sistema puede:

- ✅ Analizar imágenes satelitales de tus campos
- ✅ Calcular índices de vegetación (NDVI)
- ✅ Crear modelos 3D de tus cultivos
- ✅ Predecir riesgos y pérdidas potenciales
- ✅ Generar reportes completos para toma de decisiones

---

## 🚀 Primeros Pasos

### 1. Acceder al Sistema
1. Abre tu navegador web (Chrome, Firefox, Safari)
2. Ingresa a la dirección proporcionada por tu administrador
3. Verás la pantalla principal de AgroProRisk

### 2. Interfaz Principal
La pantalla principal se divide en:
- **Panel lateral derecho:** Controles y configuración
- **Área central:** Visualizaciones y resultados
- **Barra superior:** Título y información del sistema

---

## 📁 Carga de Datos

### Paso 1: Subir Archivos de Campo
1. En el panel lateral derecho, busca la sección **"📁 Carga de Datos"**
2. Haz clic en **"Elegir archivo"** o arrastra tu archivo
3. Tipos de archivo soportados:
   - **Archivos ZIP (.zip):** Cada archivo ZIP debe contener una imagen GeoTIFF (.tiff, .tif)
   - **DatosClimaticos.xlsx:** Archivo Excel con datos climáticos históricos
   
**⚠️ IMPORTANTE:** 
- Cada imagen satelital debe estar en un archivo .zip separado
- Asegúrate de incluir el archivo DatosClimaticos.xlsx para análisis completo

### 📋 Preparación de Archivos Requeridos

#### **🗜️ Archivos ZIP con Imágenes**
- **Formato:** Un archivo .zip por cada imagen satelital
- **Contenido:** Imagen GeoTIFF (.tiff o .tif) de tu campo
- **Nombre sugerido:** `00N_Campo_[Nombre]_[Fecha].zip` (N número de archivo)
- **Ejemplo:** `001_Campo_Maizal_2024-10-01.zip`

#### **📊 Archivo DatosClimaticos.xlsx**
- **Formato:** Archivo Excel (.xlsx)
- **Contenido:** Datos meteorológicos históricos de tu zona
- **Columnas recomendadas:**
  - Fecha
  - Temperatura máxima (°C)
  - Temperatura mínima (°C)
  - Humedad relativa (%)
  - Precipitación (mm)
  - Velocidad del viento (m/s)
- **Período:** Al menos 12 meses de datos históricos 


### Paso 2: Seleccionar Campo
1. Después de cargar, aparecerá una lista de campos disponibles
2. Selecciona el campo que deseas analizar
3. El sistema cargará automáticamente los datos

### 🔧 Configuraciones Básicas
En el panel lateral derecho encontrarás:

#### **🎨 Configuración Visual**
- **Mapa de colores:** Elige cómo ver los datos (Verde-Rojo, Viridis, etc.)
- **Transparencia:** Ajusta la opacidad de las capas
- **Resolución:** Calidad de la visualización

#### **📊 Tipo de Análisis**
- **NDVI:** Índice de vegetación (recomendado para principiantes)
- **GNDVI:** Índice verde modificado
- **SAVI:** Índice ajustado por suelo
- **EVI:** Índice de vegetación mejorado

---

## 🎮 Modos de Análisis

### 🟢 Modo 1: Visualización 3D
**¿Para qué sirve?** Ver tu campo en 3D y entender la distribución de la vegetación

**Cómo usarlo:**
1. Selecciona **"Visualización 3D"** en el panel lateral derecho
2. Configura los parámetros:
   - **Factor de escala Z:** Qué tan pronunciadas se ven las elevaciones
   - **Número de esferas:** Cantidad de puntos de datos a mostrar
3. Haz clic en **"Generar Visualización 3D"**

**¿Qué verás?**
- Un modelo 3D interactivo de tu campo
- Colores que representan la salud de la vegetación:
  - 🟢 **Verde:** Vegetación saludable
  - 🟡 **Amarillo:** Vegetación moderada
  - 🔴 **Rojo:** Vegetación con problemas

### 🔵 Modo 2: Análisis Prospectivo
**¿Para qué sirve?** Predecir cómo evolucionará tu cultivo en el tiempo

**Cómo usarlo:**
1. Selecciona **"Análisis Prospectivo"**
2. Configura:
   - **Horizonte temporal:** Cuántos meses hacia adelante predecir
   - **Escenarios:** Diferentes condiciones climáticas
3. Haz clic en **"Ejecutar Análisis Prospectivo"**

**¿Qué verás?**
- Gráficos que muestran la evolución esperada
- Predicciones mes por mes
- Zonas de riesgo identificadas

### 🟠 Modo 3: Gestión de Riesgos
**¿Para qué sirve?** Identificar riesgos potenciales y calcular pérdidas económicas

**Cómo usarlo:**
1. Selecciona **"Gestión de Riesgos"**
2. Ingresa datos económicos:
   - **Precio por tonelada:** Valor de tu cultivo
   - **Rendimiento esperado:** Toneladas por hectárea
   - **Costos de producción:** Gastos por hectárea
3. Haz clic en **"Analizar Riesgos"**

**¿Qué verás?**
- Mapas de riesgo por zonas
- Cálculos de pérdidas potenciales en dinero
- Recomendaciones de gestión

### 🟣 Modo 4: Análisis de Gases de Efecto Invernadero (GEI)
**¿Para qué sirve?** Evaluar la captura de CO2 y beneficios ambientales

**Cómo usarlo:**
1. Selecciona **"Análisis GEI"**
2. Configura parámetros ambientales
3. Haz clic en **"Procesar Datos de GEI"**

**¿Qué verás?**
- Cantidad de CO2 capturado
- Beneficios económicos ambientales
- Comparación de escenarios

---

## 📊 Interpretando los Resultados

### 🎨 Mapas de Colores
Los colores en las visualizaciones representan diferentes valores:

| Color | Significado | NDVI Valor | Estado del Cultivo |
|-------|-------------|------------|-------------------|
| 🔴 Rojo | Muy bajo | 0.0 - 0.3 | Suelo desnudo/problemas |
| 🟡 Amarillo | Bajo | 0.3 - 0.5 | Vegetación estresada |
| 🟢 Verde claro | Moderado | 0.5 - 0.7 | Vegetación saludable |
| 🟢 Verde oscuro | Alto | 0.7 - 1.0 | Vegetación muy saludable |

### 📈 Gráficos y Métricas
El sistema genera varios tipos de gráficos:

#### **Histogramas**
- Muestran la distribución de valores en tu campo
- Te ayudan a entender qué porcentaje del campo está en cada condición

#### **Gráficos de Línea**
- Muestran evolución en el tiempo
- Útiles para ver tendencias y cambios estacionales

#### **Mapas de Calor**
- Identifican zonas problemáticas
- Colores más intensos = mayor concentración de problemas

---

## 💾 Descarga de Reportes

### 📑 Tipos de Reportes Disponibles

#### **1. Reporte Excel Básico**
- Datos tabulares del análisis
- Perfecto para análisis posterior en Excel
- Incluye coordenadas y valores NDVI

#### **2. Reporte de Análisis Completo**
- Resumen ejecutivo con métricas clave
- Escenarios de riesgo detallados
- Análisis financiero y recomendaciones

#### **3. Datos HPC (High Performance Computing)**
- Análisis avanzado de riesgos
- Distribuciones probabilísticas
- Datos para modelado avanzado

### 🔄 Cómo Descargar
1. Busca el botón **"📊 Descargar..."** en la parte inferior
2. Selecciona el tipo de reporte que necesitas
3. El archivo se descargará automáticamente
4. Abre con Excel, Word o el programa correspondiente

---

## ⚙️ Configuraciones Avanzadas

### 🎛️ Panel de Control Lateral Derecho

#### **Parámetros de Visualización**
- **Altura de cámara:** Qué tan lejos ver el modelo 3D
- **Ángulo de vista:** Perspectiva de visualización
- **Intensidad de luz:** Iluminación del modelo 3D

#### **Filtros de Datos**
- **Rango de fechas:** Qué período analizar
- **Umbral NDVI:** Valores mínimos y máximos a considerar
- **Tamaño de muestra:** Cantidad de datos a procesar

#### **Opciones de Exportación**
- **Formato de archivo:** PNG, PDF, Excel
- **Resolución:** Calidad de las imágenes exportadas
- **Inclusión de metadatos:** Información adicional en archivos

---

## 🔍 Casos de Uso Prácticos

### 🌱 Caso 1: Monitoreo de Salud del Cultivo
**Objetivo:** Verificar el estado actual de mi campo de maíz

**Pasos:**
1. Cargar imagen satelital más reciente
2. Seleccionar índice NDVI
3. Usar visualización 3D con mapa de colores "RdYlGn"
4. Identificar zonas rojas (problemáticas)
5. Descargar reporte para compartir con agrónomo

### 🎯 Caso 2: Planificación de Riego
**Objetivo:** Identificar dónde necesito regar más

**Pasos:**
1. Cargar datos de campo
2. Usar análisis prospectivo con escenario seco
3. Identificar zonas de mayor riesgo de estrés hídrico
4. Planificar sistema de riego enfocado en zonas rojas/amarillas

### 💰 Caso 3: Evaluación Económica
**Objetivo:** Calcular pérdidas potenciales por sequía

**Pasos:**
1. Usar modo "Gestión de Riesgos"
2. Ingresar precios actuales del cultivo
3. Configurar escenario de sequía moderada
4. Analizar mapa de pérdidas económicas
5. Tomar decisiones sobre seguros agrícolas

### 🌍 Caso 4: Certificación Ambiental
**Objetivo:** Demostrar captura de CO2 para certificación verde

**Pasos:**
1. Usar análisis GEI
2. Comparar escenarios con y sin cultivo
3. Generar reporte de captura de CO2
4. Descargar certificado para presentar a certificadora

---

## ❓ Preguntas Frecuentes (FAQ)

### 🔧 Problemas Técnicos

**P: El sistema no carga mis archivos**
**R:** Verifica que:
- Los archivos ZIP sean menores a 100MB cada uno
- Cada ZIP contenga una imagen GeoTIFF (.tiff, .tif) válida
- Incluyas el archivo DatosClimaticos.xlsx con datos del clima
- Las imágenes tengan proyección geográfica válida (WGS84 recomendado)

**P: La visualización 3D se ve muy lenta**
**R:** 
- Reduce el número de esferas a 1000 o menos
- Baja la resolución de visualización
- Cierra otras pestañas del navegador

**P: Los colores no se ven bien**
**R:**
- Prueba diferentes mapas de colores en configuración
- Ajusta el contraste en "Configuración Visual"
- Verifica que tu monitor esté calibrado correctamente

### 📊 Interpretación de Datos

**P: ¿Qué valor NDVI es bueno para mi cultivo?**
**R:** Depende del tipo de cultivo:
- **Cereales (maíz, trigo):** 0.6-0.8 en crecimiento activo
- **Hortalizas:** 0.5-0.7 típicamente
- **Frutales:** 0.7-0.9 en época de crecimiento
- **Pastos:** 0.4-0.7 según estación

**P: ¿Con qué frecuencia debo hacer análisis?**
**R:**
- **Semanal:** Durante época crítica de crecimiento
- **Quincenal:** Durante crecimiento normal
- **Mensual:** Para monitoreo de rutina
- **Después de eventos:** Lluvia intensa, sequía, heladas

**P: ¿Cómo interpretar las predicciones?**
**R:**
- **Verde:** Bajo riesgo, continúa manejo actual
- **Amarillo:** Riesgo moderado, monitorea de cerca
- **Rojo:** Alto riesgo, considera acciones correctivas

### 💼 Uso Empresarial

**P: ¿Puedo usar esto para múltiples campos?**
**R:** Sí, puedes analizar diferentes campos cargando sus respectivos archivos. Recomendamos organizarlos por:
- Nombre del campo
- Fecha del análisis
- Tipo de cultivo

**P: ¿Cómo comparto resultados con mi equipo?**
**R:**
- Descarga reportes Excel para análisis detallado
- Toma capturas de pantalla de visualizaciones
- Usa reportes PDF para presentaciones ejecutivas

**P: ¿Es seguro subir mis datos?**
**R:** El sistema está diseñado con seguridad estándar. Para datos altamente sensibles, consulta con tu administrador sobre políticas de privacidad específicas.

---

## 🆘 Solución de Problemas

### ⚠️ Problemas Comunes y Soluciones

#### **Error: "Archivo no reconocido"**
**Causa:** Formato de archivo incorrecto
**Solución:**
1. Verifica que cada archivo sea un .zip conteniendo imágenes .tiff/.tif
2. Confirma que incluyas el archivo DatosClimaticos.xlsx
3. Asegúrate de que las imágenes tengan georeferenciación válida
4. Convierte el archivo si es necesario

#### **Error: "Datos insuficientes"**
**Causa:** Área muy pequeña o datos incompletos
**Solución:**
1. Verifica que el área sea mayor a 1 hectárea
2. Confirma que la imagen tenga datos válidos
3. Prueba con un archivo diferente

#### **Visualización en blanco**
**Causa:** Parámetros de visualización incorrectos
**Solución:**
1. Reinicia configuración a valores por defecto
2. Ajusta rango de valores NDVI
3. Cambia mapa de colores

#### **Lentitud en el sistema**
**Causa:** Sobrecarga de datos o hardware limitado
**Solución:**
1. Reduce resolución de análisis
2. Procesa áreas más pequeñas
3. Cierra aplicaciones innecesarias

### 🔄 Reinicio del Sistema
Si experimentas problemas persistentes:
1. Recarga la página (F5 o Ctrl+R)
2. Limpia caché del navegador
3. Contacta soporte técnico si persiste

---

## 📞 Soporte y Contacto

### 🎓 Información Académica
- **Institución:** Universidad EAFIT
- **Investigador Principal:** Alejandro Puerta
- **Proyecto:** Análisis de Riesgo en Cultivos mediante Imágenes Satelitales

### 📚 Recursos Adicionales
- Manual técnico disponible en la carpeta `/docs`
- Videos tutoriales (si están disponibles)
- Documentación API para desarrolladores

### 🔧 Soporte Técnico
Para asistencia técnica:
1. Documenta el problema específico
2. Incluye capturas de pantalla si es posible
3. Especifica qué archivo estabas procesando
4. Contacta al administrador del sistema

---

## 📋 Lista de Verificación Rápida

### ✅ Antes de Empezar
- [ ] Tengo archivos ZIP con imágenes GeoTIFF (.tiff/.tif) listos
- [ ] Cada archivo ZIP es menor a 100MB
- [ ] Tengo el archivo DatosClimaticos.xlsx preparado
- [ ] Conozco el tipo de cultivo que estoy analizando
- [ ] Tengo datos económicos si voy a hacer análisis de riesgo

### ✅ Durante el Análisis
- [ ] Seleccioné el índice de vegetación apropiado
- [ ] Configuré parámetros según mi objetivo
- [ ] Verifiqué que los colores se vean correctamente
- [ ] Entiendo lo que muestran los resultados

### ✅ Al Finalizar
- [ ] Descargué los reportes necesarios
- [ ] Guardé capturas de pantalla importantes
- [ ] Documenté hallazgos principales
- [ ] Planifiqué próximos análisis o acciones

---

## 🎉 ¡Felicitaciones!

Has completado el manual de usuario de AgroProRisk. Con esta herramienta podrás:

- 🎯 **Tomar mejores decisiones** basadas en datos objetivos
- 📈 **Optimizar rendimientos** identificando problemas temprano
- 💰 **Reducir pérdidas** con análisis predictivo
- 🌍 **Contribuir al ambiente** monitoreando captura de CO2

**¡Tu agricultura ahora es más inteligente y sostenible!**

---

*© 2025 Universidad EAFIT - Alejandro Puerta. Todos los derechos reservados.*
*Este manual es parte del proyecto de investigación en análisis de riesgo agrícola mediante imágenes satelitales.*