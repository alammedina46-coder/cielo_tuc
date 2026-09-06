# PLAN — CIELO·TUC

> Sistema de Predicción Climática con Inteligencia Artificial para la Provincia de Tucumán, Argentina.
> Documento vivo: estado actual, brechas y plan de fases para completar el proyecto.

---

## 1. Qué es CIELO·TUC

CIELO·TUC es un sistema hiperlocal de predicción climática con IA diseñado exclusivamente para Tucumán, que combina:

- **Investigación** (`cielo-tuc-investigacion-final.docx` / `.md`): fundamentos científicos, análisis económico y la visión de 13 capítulos con 24 tablas.
- **Backend** (`cielotuc-backend/cielotuc/`): API FastAPI + modelo CNN-LSTM + pipeline de datos + tareas programadas + integración con FLOOD·TUC.
- **Frontend (pendiente)**: dos interfaces — ciudadanos y gobierno.

La promesa central: superar la precisión de SMN (~67%) y de los servicios globales (~72%) en el territorio tucumano, llegando a >91% en pronósticos de corto plazo.

---

## 2. Estado actual — ~45%

Comparación contra el roadmap de la investigación (fases 0–7):

| Fase | Estado | % |
|---|---|---|
| 0 — Setup (infraestructura + BD) | Código, docker-compose y seed listos. **Faltan Alembic y .env real** | ~90% |
| 1 — MVP dashboard | **Backend 100%; frontend 0%** | ~55% |
| 2 — Tiempo real | Tasks Celery escritas. **Sin Windy API, sin EEAOC real, sin desplegar** | ~45% |
| 3 — Modelo IA v1 | Arquitectura + trainer listos. **Modelo no entrenado ni validado; labels heurísticos** | ~55% |
| 4 — Reentrenamiento | Task mensual definida. Depende de datos reales | ~40% |
| 5 — Gobierno + FLOOD·TUC | Service y endpoints listos. **Sin frontend, sin Twilio, sin auth** | ~30% |
| 6 — Validación 6 meses | Nada aún | 0% |
| 7 — Escalabilidad NOA | Nada aún | 0% |

---

## 3. Inventario existente

### Backend (completo en código)
- **API FastAPI** — 13 endpoints:
  - `GET /health`
  - `GET /api/v1/zones/` y `/api/v1/zones/{id}` (17 departamentos)
  - `GET /api/v1/forecast/{zone_id}` (condiciones actuales + 12h + 7 días)
  - `GET /api/v1/forecast/{zone_id}/zonda` (índice de riesgo)
  - `GET /api/v1/sensors/status`, `POST /api/v1/sensors/readings`
  - `POST /api/v1/alerts/flood`, `GET /api/v1/alerts/flood/history`
  - `GET /api/v1/model/metrics`, `GET /api/v1/model/comparison`, `POST /api/v1/model/retrain`
- **Modelo CNN-LSTM** (`app/ml/models/cnn_lstm.py`): 34 features, ventana 72 hs, 6 horizontes (3/6/12/24/48/168h), 7 salidas por horizonte (lluvia, precipitación, temperatura, viento, Zonda, tormenta, granizo), atención temporal, loss multi-task.
- **Pipeline de datos** (`app/ml/pipeline/data_pipeline.py`): SMN, NASA POWER, NASA GPM, ERA5/CDS + features derivadas (CAPE proxy, K-index, Zonda proxy, features cíclicas sin/cos).
- **Entrenamiento** (`app/ml/training/trainer.py`): MLflow, TimeSeriesSplit, early stopping, checkpoints.
- **Tareas Celery** (`app/services/tasks.py`): fetch cada 15 min, predicciones cada 30 min, validación cada 30 min, retrain mensual (día 1, 02:00).
- **FLOOD·TUC** (`app/services/flood_alert_service.py`): alerta HTTP automática cuando lluvia ≥ 70mm/h.
- **PostgreSQL + PostGIS**: 7 tablas ORM (zones, weather_stations, sensor_readings, weather_events, ai_predictions, model_versions, flood_alerts).
- **Docker Compose**: db (PostGIS), redis, api, worker, beat, mlflow.
- **Scripts**: `init_db.py` (seed 17 departamentos + 12 estaciones) y `train_initial_model.py` (NASA POWER/ERA5 + datos sintéticos + entrenamiento).
- **Tests**: 12 unit (ML) + 11 integración (API).

---

## 4. Gaps detectados

| Gap | Detalle | Impacto |
|---|---|---|
| **Frontend** | No existe ninguna carpeta React/Vite | Sin interfaz ciudadana ni de gobierno |
| **Windy API** | Mencionada en la investigación, ausente en el código | Pierde el diferenciador clave (ECMWF/GFS multi-nivel para el Zonda) |
| **Entrenamiento real** | El modelo nunca se entrenó ni validó; labels de Zonda/tormenta/granizo son heurísticos | Precisión real desconocida |
| **Alembic** | En requirements, sin carpeta ni migraciones | Sin versionado de esquema |
| **Autenticación** | python-jose/passlib en requirements, sin implementar | Panel gobierno sin control de acceso |
| **Notificaciones** | Twilio en requirements, sin implementar | Sin SMS/WhatsApp |
| **Deploy** | Sin despliegue, sin CI/CD, sin dominio | No accesible públicamente |
| **EEAOC** | Solo seed de estaciones; sin fetch real | Datos agro perdidos |

---

## 5. Fases pendientes

### Fase A — Cerrar el backend (prioridad: ALTA)

**Objetivo:** dejar el backend operando end-to-end y con el diferenciador Windy integrado.

**Entregables:**
- [ ] Nuevo `windy_client.py` — Point Forecast API (ECMWF + GFS, niveles superficie/850/700/500/300 hPa) para los pronósticos en altura y el predictor del Zonda.
- [ ] Integrar Windy en `data_pipeline.py` (fetch + merge con SMN/NASA) y en `tasks.py`.
- [ ] Migraciones Alembic iniciales (`alembic init` + revisión de las 7 tablas).
- [ ] `.env` real de desarrollo (SECRET_KEY generada, credenciales locales).
- [ ] Levantar infra: `docker compose up -d`, seed, arrancar Celery worker + beat.
- [ ] Mejorar labels de entrenamiento: reemplazar heurísticas por eventos reales (weather_events) + los `_label_*` sintéticos ya generados.
- [ ] Correr suite completa de tests (unit + integración) y dejar verde.

**Validación:** API respondiendo en `localhost:8000/docs`, tasks Celery ejecutando los 4 schedulers, tests en verde.

---

### Fase B — Frontend ciudadano (prioridad: ALTA)

**Objetivo:** la vista para el ciudadano tucumano, 100% responsive y en español.

**Entregables:**
- [ ] Scaffold React 18 + Vite + TanStack Query + Tailwind (`cielotuc-frontend/`).
- [ ] Hero Weather (temperatura en 96 pt, badge de confianza del modelo).
- [ ] Pronóstico horario — próximas 12 horas (barras de probabilidad verde→amarillo→naranja→rojo).
- [ ] Comparación con pronósticos oficiales (SMN vs Weather.com, últimos 30 días).
- [ ] Mapa de microclimas por zona (React Leaflet + tiles dark).
- [ ] Monitor del viento Zonda (gauge semicircular 0–100).

**Validación:** navegación completa en localhost:5173 conectada al backend, pantallas cargando datos reales.

---

### Fase C — Frontend gobierno (prioridad: MEDIA)

**Objetivo:** el panel técnico para municipios y organismos provinciales.

**Entregables:**
- [ ] KPI strip de seis indicadores (incluye "alertas enviadas a FLOOD·TUC este mes").
- [ ] Tabla de comparación histórica con SMN (casos anticipados + horas de antelación).
- [ ] Gauges de eventos extremos (el de mayor riesgo parpadea).
- [ ] Panel de alertas FLOOD·TUC (historial, delivery, severidad).
- [ ] Botón de retrain manual del modelo (POST /model/retrain).

**Validación:** acceso por rol, datos de la API de gobierno, responsive.

---

### Fase D — Modelo real + validación (prioridad: ALTA)

**Objetivo:** que el 91%+ no sea una proyección sino una métrica medida.

**Entregables:**
- [ ] Entrenar v1.0 con datos reales (NASA POWER + ERA5 si hay credenciales) + aumento sintético.
- [ ] Backtesting con TimeSeriesSplit y registro en MLflow.
- [ ] Métricas por horizonte: precisión de lluvia, RMSE de temperatura, falsos positivos.
- [ ] Línea base comparativa: SMN y Weather.com para el mismo período.
- [ ] Ciclo mensual de réentrenamiento verificado (swap de versión activa automático).

**Validación:** precisión >76% en backtesting (hito v1), >85% a los 6 meses de datos reales, >91% a los 18 meses (v3.2), >94% a los 24 meses (v4.0).

---

### Fase E — Notificaciones y seguridad (prioridad: MEDIA)

**Objetivo:** alertas proactivas y acceso controlado.

**Entregables:**
- [ ] Autenticación JWT con roles: `citizen` y `government`.
- [ ] Endpoints de registro/login + middleware de protección.
- [ ] Twilio SMS + WhatsApp para alertas de riesgo (Zonda, tormenta, FLOOD·TUC).
- [ ] Preferencias de suscripción por zona y canal.
- [ ] Rate limiting y hardening básico (CORS estricto, validación de inputs).

**Validación:** flujo login→panel gobierno protegido; alerta SMS/WhatsApp disparada por un evento de prueba.

---

### Fase F — Puesta en producción (prioridad: MEDIA)

**Objetivo:** CIELO·TUC accesible y monitoreable en producción.

**Entregables:**
- [ ] Deploy en Railway (API + worker + beat + Redis + PostGIS + MLflow).
- [ ] Dominio propio + TLS.
- [ ] CI/CD (GitHub Actions: lint, tests, build, deploy).
- [ ] Monitoreo (health checks, logs centralizados, métricas).
- [ ] Conexión en vivo con FLOOD·TUC real (URL + API key en producción).

**Validación:** URL pública funcional, despliegue automático desde `main`, alertas en vivo.

---

### Fase G — Escalabilidad al NOA (prioridad: BAJA)

**Objetivo:** expandir el sistema más allá de Tucumán.

**Entregables:**
- [ ] Salta (6 meses post-TUC): datos de la red hidrometeorológica del Bermejo.
- [ ] Jujuy (6 meses post-Salta): estaciones 3.000–4.500 msnm.
- [ ] Catamarca (9 meses post-Jujuy): temperatura extrema, datos CONICET.
- [ ] Santiago del Estero (12 meses post-Catamarca): llanura de inundación + FLOOD·TUC NOA.
- [ ] La Rioja (paralelo a Catamarca): reutilización casi directa del oeste tucumano.

**Validación:** red de 200 estaciones en el NOA; primer dataset meteorológico completo de la región andina argentina.

---

## 6. Métricas objetivo

| Métrica | Hoy | 6 meses | 12 meses | 18 meses | 24 meses |
|---|---|---|---|---|---|
| Precisión pronóstico <12h | — (sin medir) | >85% | >89% | >91,4% | >94% |
| Comparación vs SMN (67%) | — | +15 pts | +20 pts | +22 pts | +25 pts |
| Estaciones activas | 12 (seed) | 40 | 60 | 100 | 150+ |
| Alertas FLOOD·TUC enviadas | 0 | registradas | validadas | reporte público | reporte público |
| Provincias NOA | 1 | 1 | 2 | 3 | 5 |

---

## 7. Orden recomendado de ejecución

```
Fase A → Fase B → Fase D (en paralelo con B) → Fase C → Fase E → Fase F → Fase G
```

La Fase D (modelo real) puede correr en paralelo a la B (frontend ciudadano) porque no comparten dependencias directas más allá de la API.

---

*CIELO·TUC · Sistema de Predicción Climática con IA · Tucumán, Argentina · 2026*
