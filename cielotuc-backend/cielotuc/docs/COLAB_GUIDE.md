# CIELO·TUC — Entrenamiento en Google Colab (GPU gratis)

## Por qué Colab?
- **GPU gratis** (NVIDIA T4) — entrena en ~5 min lo que en CPU tarda 1+ hora
- **15 años de datos** + 2000 eventos sintéticos
- **Modelo v3.0** más grande (128 hidden, 3 LSTM layers, residual connections)
- **Mixed precision** — 2x más rápido en GPU

## Paso a paso

### 1. Abrir Colab
Ir a [https://colab.research.google.com](https://colab.research.google.com)

### 2. Activar GPU
- Menú: **Runtime** → **Change runtime type**
- Seleccionar: **T4 GPU**
- Guardar

### 3. Subir el script
Opción A (recomendado): Subir `scripts/train_v2_colab.py` desde tu computadora
- Menú: **File** → **Upload session storage**
- Seleccionar el archivo

Opción B: Copiar desde GitHub
```python
!wget https://raw.githubusercontent.com/alammedina46-coder/cielo_tuc/main/cielotuc-backend/cielotuc/scripts/train_v2_colab.py
```

### 4. Ejecutar
```python
!python train_v2_colab.py --years 15 --max-epochs 150
```

Tiempo estimado: **~5 minutos** en T4 GPU.

### 5. Descargar modelo entrenado
Al finalizar, el script muestra instrucciones de descarga:
```python
from google.colab import files
files.download('models/cielotuc_v3.0-YYYYMMDD.pt')
files.download('models/metrics_v3.0-YYYYMMDD.json')
```

### 6. Subir al proyecto
Copiar los archivos descargados a `cielotuc-backend/cielotuc/models/` y pushear a GitHub.

## Mejoras v3.0 sobre v2.0

| Aspecto | v2.0 (CPU) | v3.0 (Colab GPU) |
|---------|-----------|------------------|
| Datos | 10 años NASA POWER | 15 años NASA POWER |
| Sintéticos | 1000 | 2000 |
| Modelo | 168K params | ~400K params |
| Lookback | 24h | 24h |
| Loss | BCE + MAE | Focal BCE + Huber |
| Scheduler | Cosine | Warmup + Cosine |
| Grad accum | No | Yes (effective 1024) |
| Mixed precision | No | Yes (AMP) |
| Residual connections | No | Yes |
| SE attention | No | Yes |

## Resultados esperados

- **v2.0 (actual)**: 75.87% composite, 70.4% rain, 13.3°C RMSE
- **v3.0 (Colab)**: ~78-82% composite, ~73-76% rain, ~11-12°C RMSE
