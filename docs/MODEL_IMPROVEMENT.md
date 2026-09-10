# CIELO·TUC — Seguimiento de Mejora del Modelo ML

## Meta final (según investigación)

| Métrica | Objetivo | Fuente |
|---------|----------|--------|
| Precisión general (<12h) | **>91%** (objetivo >94%) | Tabla 1, sección 3.1 |
| Precisión v1.0 (backtesting) | **>76%** | Roadmap fase 3 |
| Precisión v6.0 (24 meses datos reales) | **>94%** | Roadmap fase 6 |
| Ventaja sobre SMN (~67%) | **>24pp de diferencia** | Tabla 1, sección 3.1 |
| Ventaja sobre Weather.com (~72%) | **>19pp de diferencia** | Tabla 1, sección 3.1 |

## Roadmap de mejora continua (investigación vs realidad)

| Versión | Datos de entrenamiento | Precisión proyectada | Precisión real | Estado |
|---------|----------------------|---------------------|---------------|--------|
| v1.0 | Históricos únicamente | ~76% | 55.5% | Entrenado (CPU) |
| v2.0 | Históricos + mejoras | ~81% | 75.9% | Entrenado (CPU) |
| v3.0 | 15yr NASA + 2000 synth | ~85% | 71.0% | Entrenado (GPU) |
| v4.0 | 5 zonas + CV temporal | ~89% | 91.8%* | Entrenado (GPU) |
| v5.0 | 10 zonas + 3000 synth | ~91% | **83.8%** | **ACTUAL** |
| v6.0 | Datos reales + retrain | >94% | — | Pendiente |

*\*v4.0 tuvo inflado el score por data leakage (solo evaluó Capital con synthetic distribuido desigual)*

## Métricas detalladas v5.0 (actual)

| Métrica | Valor | Brecha vs objetivo |
|---------|-------|-------------------|
| Composite accuracy (weighted) | **83.8%** | -7.2pp vs 91% |
| Rain accuracy (avg) | **67.1%** | -24pp vs 91% |
| Temp RMSE | **7.3°C** | — |

### Por horizonte

| Horizonte | Composite | Rain Acc | Temp RMSE | Estado |
|-----------|-----------|----------|-----------|--------|
| 3h | 90.3% | 78.9% | 5.1°C | Casi objetivo |
| 6h | 89.1% | 77.3% | 5.9°C | Casi objetivo |
| 12h | 86.6% | 73.1% | 6.5°C | Necesita mejora |
| 24h | 79.2% | 62.4% | 8.1°C | Débil |
| 48h | 72.5% | 56.3% | 9.2°C | Muy débil |
| 168h | 71.2% | 55.4% | 9.3°C | Muy débil |

### Por zona

| Zona | Composite | Estado |
|------|-----------|--------|
| Lules | 84.8% | Mejor |
| Capital | 84.2% | OK |
| Capital Norte | 84.0% | OK |
| Yerba Buena | 83.9% | OK |
| Cruz Alta | 83.8% | OK |
| Tafi Viejo | 84.1% | OK |
| Concepcion | 83.5% | OK |
| Bella Vista | 83.3% | OK |
| Monteros | 83.1% | Débil |
| Chicligasta | 82.8% | Débil |

## Diagnóstico: Por qué no llega a 91%

### 1. Falta de datos reales
- **Problema**: El modelo se entrenó 100% con datos sintéticos para eventos extremos. Los sin-wave events no capturan la dinámica real de tormentas tucumanas.
- **Solución**: Necesita 6-12 meses de datos reales de estaciones IoT + reentrenamiento mensual.

### 2. Horizontes largos (24h+) degradan el promedio
- **Problema**: 3h=90.3% pero 168h=71.2%. El promedio ponderado baja a 83.8%.
- **Solución**: Entrenar horizontes independientes o usar loss con pesos más agresivos para corto plazo.

### 3. Precipitación sub-predicha en lluvias moderadas
- **Problema**: Rain accuracy promedio 67.1% — especialmente bajo en 24h+ (56-62%).
- **Solución**: Más datos de precipitación real, augmentación de training data con eventos de lluvia moderada (10-30mm).

### 4. Eventos extremos: zonda/hail precision=0 en horizontes largos
- **Problema**: A 48h y 168h, la precisión de detección de Zonda y granizo es 0%.
- **Solución**: Datos reales de Zonda son escasos. Necesita más eventos sintéticos calibrados con datos de ERA5 upper-air.

### 5. Arquitectura puede no ser suficiente
- **Problema**: 232K params puede ser insuficiente para capturar la complejidad de 10 zonas × 6 horizontes × 7 targets.
- **Solución**: Evaluar arquitecturas más profundas o attention mechanisms multi-head.

## Plan de mejora — Próximos pasos

### Fase 1: Mejoras al modelo actual (sin datos reales)
- [ ] Evaluar horizontes independientes (modelos separados para 3h, 6h, 24h, 168h)
- [ ] Experimentar con más data augmentation para lluvia moderada
- [ ] Agregar más zonas de entrenamiento (20+ en vez de 10)
- [ ] Probar scheduler con warmup más largo + cosine decay
- [ ] Evaluar ensemble de modelos v2.0 + v5.0

### Fase 2: Datos reales (cuando haya estaciones IoT)
- [ ] Recolectar 3 meses de readings reales
- [ ] Reentrenar con mix 80% real + 20% sintético
- [ ] Comparar v5.0 vs v6.0 en datos reales

### Fase 3: Entrenamiento continuo
- [ ] Pipeline mensual de reentrenamiento automático
- [ ] Validación cruzada con datos reales por zona
- [ ] Target: >85% a los 6 meses, >91% a los 12 meses

## Archivos de modelos entrenados

| Archivo | Tamaño | Métricas |
|---------|--------|----------|
| `cielotuc_v1.0-20260905.pt` | 1.7MB | 55.5% composite |
| `cielotuc_v2.0-20260907.pt` | 700KB | 75.9% composite |
| `cielotuc_v3.0-20260909.pt` | 2.1MB | 71.0% composite |
| `cielotuc_v4.0-20260909.pt` | 1.7MB | 91.8%* (inflado) |
| `cielotuc_v5.0-20260910.pt` | ~2MB | 83.8% composite **ACTUAL** |

**POLÍTICA: No se sube el `.pt` a producción (Render/GitHub) hasta que el modelo alcance >91% composite accuracy en evaluación con datos sintéticos balanceados, o hasta que se tenga al menos 3 meses de datos reales para validación.**

---
*Última actualización: 2026-09-10*
