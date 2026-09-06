# Neon Database Setup — CIELO·TUC

Guide para crear y configurar una base PostgreSQL + PostGIS en **Neon** (free tier) para desarrollo del backend.

## 1. Crear la cuenta

1. Andá a https://neon.tech y hacé clic en **"Sign up"**.
2. Podés registrarte con email o con cuenta de **GitHub**/Google.
3. Cuando entres al dashboard, elegí el plan **"Free Tier"** (no pide tarjeta de crédito).

## 2. Crear el proyecto

1. Clic en **"Create a project"** (o **"New Project"**).
2. **Name**: `cielotuc` (o el que quieras).
3. **Database name**: `cielotuc`.
4. **Region**: elegí la más cercana a Argentina. Neon suele ofrecer `sa-east-1` (**São Paulo**), que es la mejor opción para Tucumán. Si no aparece, dejá la default.
5. Clic en **"Create project"**.

## 3. Obtener la connection string

1. En la pantalla siguiente, te muestra las **connection strings**. Buscá la que dice **"Connection string"** (no la pooled).
2. Copiá la que empieza con `postgresql://...`. La estructura es:

```
postgresql://neondb_owner:PASSWORD@ep-xxxxx.region.aws.neon.tech/neondb?sslmode=require
```

3. **Importante**: guardá la password porque se muestra solo una vez.

## 4. Habilitar PostGIS

En el SQL Editor de Neon (menú izquierdo > "SQL Editor"), ejecutá:

```sql
CREATE EXTENSION IF NOT EXISTS postgis;
```

Esto es necesario porque nuestro schema usa columnas geográficas y funciones espaciales.

## 5. Pegarle la URL al .env

Una vez tengas la connection string, la Copiás y me la pasás. Yo la adapto al `.env` con los dos formatos que necesita el proyecto:

```
# URL async (para SQLAlchemy async)
DATABASE_URL=postgresql+asyncpg://USUARIO:CLAVE@ep-xxxxx.region.aws.neon.tech/cielotuc?sslmode=require

# URL sync (para Alembic / seed / MLflow)
DATABASE_URL_SYNC=postgresql://USUARIO:CLAVE@ep-xxxxx.region.aws.neon.tech/cielotuc?sslmode=require
```

## 6. Verificar conexión

Yo corro automáticamente:

```bash
# Test de conexión sync
alembic upgrade head

# Seed de datos iniciales (17 departamentos + estaciones)
python scripts/seed_departments.py

# Tests de integración
pytest tests/integration/ -v
```

## Notas

- **Free tier**: 0.5 GB de almacenamiento, 100 horas de cómputo/mes.
- **Auto-pausa**: la compute se pausa tras inactividad, pero despierta sola al recibir conexión (~1s).
- **SSL**: Neon requiere `?sslmode=require` en todas las conexiones.
- **Branching**: Neon permite crear "branches" de la DB (útil para testear migraciones sin afectar dev).

## Referencia

- Docs Neon: https://neon.tech/docs
- PostGIS en Neon: https://neon.tech/docs/extensions/postgis
- Alembic + asyncpg: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
