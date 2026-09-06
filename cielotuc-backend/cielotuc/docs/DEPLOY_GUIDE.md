# CIELO·TUC — Deploy 100% Gratuito (o casi)

## Stack

| Componente | Servicio | Costo |
|-----------|----------|-------|
| **Backend** | Render (Free tier) | $0 |
| **Frontend** | Vercel (Hobby) | $0 |
| **PostgreSQL** | Neon (Free tier) | $0 |
| **Dominio** | .com.ar (NIC.ar) | ~$1-2 USD/año |

**Sin Redis, sin Celery, sin Docker en producción.** Las tareas background usan APScheduler integrado en el proceso de FastAPI.

---

## Paso 1: Crear base de datos Neon (5 min)

1. Andá a **https://neon.tech** y creá cuenta (con GitHub)
2. Creá un proyecto nuevo:
   - **Database name:** `cielotuc`
   - **Region:** AWS US East (Virginia) o South America (São Paulo)
3. Copiá el **connection string** — algo como:
   ```
   postgresql://neondb_owner:xxxx@ep-yyy.us-east-2.aws.neon.tech/cielotuc?sslmode=require
   ```
4. Andá a SQL Editor en Neon Dashboard y ejecutá:
   ```sql
   CREATE EXTENSION IF NOT EXISTS postgis;
   ```
5. Exportá las variables:
   ```bash
   # Async (para la app)
   export DATABASE_URL="postgresql+asyncpg://neondb_owner:xxxx@ep-yyy.neon.tech/cielotuc?sslmode=require"
   # Sync (para Alembic y seed)
   export DATABASE_URL_SYNC="postgresql://neondb_owner:xxxx@ep-yyy.neon.tech/cielotuc?sslmode=require"
   ```

---

## Paso 2: Poblar la DB con seed (2 min)

```bash
cd cielotuc-backend/cielotuc
pip install -r requirements.txt
python -c "from alembic.config import Config; from alembic import command; command.upgrade(Config('alembic.ini'), 'head')"
python scripts/seed_neon.py
```

Esto crea 17 zonas climáticas de Tucumán + 11 estaciones meteorológicas.

---

## Paso 3: Subir backend a Render (10 min)

1. Andá a **https://render.com** y creá cuenta (con GitHub)
2. **New > Web Service**
3. Conectá tu repo GitHub (`cielo_tuc`)
4. Configurá:
   - **Name:** `cielotuc-backend`
   - **Runtime:** Python 3
   - **Build Command:** `cd cielotuc-backend/cielotuc && pip install -r requirements.txt`
   - **Start Command:** `cd cielotuc-backend/cielotuc && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
5. En **Environment Variables** agregá:
   | Key | Value |
   |-----|-------|
   | `DATABASE_URL` | `postgresql+asyncpg://neondb_owner:xxxx@ep-yyy.neon.tech/cielotuc?sslmode=require` |
   | `DATABASE_URL_SYNC` | `postgresql://neondb_owner:xxxx@ep-yyy.neon.tech/cielotuc?sslmode=require` |
   | `SECRET_KEY` | (click Generate) |
   | `DEBUG` | `false` |
   | `SCHEDULER_ENABLED` | `true` |
   | `CORS_ORIGINS` | `https://cielotuc.vercel.app` |
   | `MLFLOW_TRACKING_URI` | `https://dummy` (MLflow no se usa en deploy) |
6. Click **Create Web Service**
7. Esperá ~5 min a que haga build y deploy
8. Tu backend va a estar en: `https://cielotuc-backend.onrender.com`
9. Verificá: `https://cielotuc-backend.onrender.com/health`

**Nota:** El free tier de Render duerme después de 15 min de inactividad. La primera request después de dormir toma ~30s en despertar.

---

## Paso 4: Subir frontend a Vercel (5 min)

1. Andá a **https://vercel.com** y creá cuenta (con GitHub)
2. **New Project** > importá tu repo
3. Configurá:
   - **Framework:** Vite
   - **Root Directory:** `cielotuc-frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. En **Environment Variables** agregá:
   | Key | Value |
   |-----|-------|
   | `VITE_API_BASE_URL` | `https://cielotuc-backend.onrender.com` |
5. Click **Deploy**
6. Tu frontend va a estar en: `https://cielotuc.vercel.app`

---

## Paso 5: Dominio .com.ar (opcional, ~$1-2 USD/año)

1. Andá a **https://nic.ar** (registro de dominios .com.ar)
2. Buscá disponibilidad: `cielotuc.com.ar`
3. Registrá (~$800-1500 ARS/año)
4. En Vercel, andá a **Settings > Domains** y agregá `cielotuc.com.ar`
5. Vercel te da un CNAME o A record — configurá eso en NIC.ar
6. SSL automático por Vercel

---

## Paso 6: Asociar dominio al backend (opcional)

Si querés que `api.cielotuc.com.ar` apunte al backend:

1. En NIC.ar configurá: `api` → CNAME → `cielotuc-backend.onrender.com`
2. Actualizá `CORS_ORIGINS` en Render: `https://cielotuc.com.ar,https://www.cielotuc.com.ar`
3. Actualizá `VITE_API_BASE_URL` en Vercel: `https://api.cielotuc.com.ar`

---

## Verificación post-deploy

```bash
# Backend health check
curl https://cielotuc-backend.onrender.com/health

# Zonas (debería devolver 17)
curl https://cielotuc-backend.onrender.com/api/v1/zones/

# Documentación Swagger
open https://cielotuc-backend.onrender.com/docs
```

---

## Limitaciones del Free tier

| Servicio | Limitación | Impacto |
|----------|-----------|---------|
| Render Free | Duermen después de 15 min | Primera request toma ~30s |
| Neon Free | 0.5 GB storage, 191h compute/mes | Suficiente para MVP |
| Vercel Hobby | 100 GB bandwidth/mes | Más que suficiente |
| NIC.ar | No hay límites | Solo costo anual |

---

## Variables de entorno completas

```env
# Base de datos (Neon)
DATABASE_URL=postgresql+asyncpg://user:pass@host.neon.tech/cielotuc?sslmode=require
DATABASE_URL_SYNC=postgresql://user:pass@host.neon.tech/cielotuc?sslmode=require

# Seguridad
SECRET_KEY=tu-clave-secreta-generada

# App
DEBUG=false
SCHEDULER_ENABLED=true

# CORS (Vercel frontend)
CORS_ORIGINS=https://cielotuc.vercel.app

# APIs externas (opcional - la app funciona sin)
WINDY_API_KEY=
SMN_API_URL=https://ws.smn.gob.ar/map_items/weather
NASA_POWER_BASE_URL=https://power.larc.nasa.gov/api/temporal

# Twilio (opcional)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_PHONE=
TWILIO_WHATSAPP_FROM=
```
