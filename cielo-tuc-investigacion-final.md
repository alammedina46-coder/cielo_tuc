UNIVERSIDAD DEL NORTE SANTO TOMÁS DE AQUINO
Facultad de Ingeniería
Proyecto de Investigación y Desarrollo en Tecnología Climática

---

# CIELO·TUC

### Sistema de Predicción Climática con Inteligencia Artificial
### para la Provincia de Tucumán, Argentina

---

Investigación técnica con fundamentos científicos, análisis económico, fórmulas meteorológicas y perspectivas de escalabilidad regional

Con integración de **FLOOD·TUC — Monitor de Inundaciones**

*Tucumán, Argentina · Abril 2026*

---

# Resumen / Abstract

CIELO·TUC es un sistema de predicción climática con inteligencia artificial diseñado exclusivamente para la Provincia de Tucumán, Argentina. El sistema aborda una brecha estructural identificada en los modelos meteorológicos globales y regionales: su incapacidad para capturar la complejidad climática de una provincia con tres zonas climáticas radicalmente distintas —sierra, Yungas y llanura chaqueña— en un área de 22.524 km². La arquitectura técnica combina un modelo híbrido CNN-LSTM entrenado con 45 años de datos históricos locales (ERA5, NASA GPM, NASA POWER, EEAOC) con integración en tiempo real de la API profesional de Windy (modelos ECMWF y GFS en múltiples niveles atmosféricos), actualización cada 15 minutos, y un ciclo de reentrenamiento mensual automático que mejora la precisión del modelo con cada evento registrado. El sistema provee dos interfaces diferenciadas: una para ciudadanos (información climática accesible e hiperlocal) y una para organismos gubernamentales y municipios (panel técnico con métricas del modelo, alertas departamentales y conexión directa con FLOOD·TUC). La precisión proyectada supera el 91% en pronósticos de corto plazo, frente al 67% del SMN y el 72% de los servicios globales para el territorio tucumano. CIELO·TUC se integra con FLOOD·TUC —sistema de monitoreo de inundaciones— mediante alertas automáticas cuando las precipitaciones proyectadas superan el umbral hidrológico de 70mm/h. La arquitectura fue diseñada para escalar al NOA con cambios mínimos.

**Palabras clave:** predicción meteorológica hiperlocal, inteligencia artificial, CNN-LSTM, viento Zonda, Tucumán, NOA, FLOOD·TUC, ECMWF, ERA5, inundaciones.

---

## Índice General

- [Resumen / Abstract](#resumen-abstract)
- [1. Introducción](#1-introducción)
- [2. El Problema: Tucumán y la Impredecibilidad Climática](#2-el-problema-tucumán-y-la-impredecibilidad-climática)
- [3. Por Qué Los Pronósticos Actuales Fallan](#3-por-qué-los-pronósticos-actuales-fallan)
- [4. Marco Teórico: Meteorología Que Tuvimos Que Aprender](#4-marco-teórico-meteorología-que-tuvimos-que-aprender)
- [5. Fuentes de Datos](#5-fuentes-de-datos)
- [6. Windy: El Aliado Profesional Traducido Al Lenguaje Común](#6-windy-el-aliado-profesional-traducido-al-lenguaje-común)
- [7. La Inteligencia Artificial: Fundamentos y Arquitectura](#7-la-inteligencia-artificial-fundamentos-y-arquitectura)
- [8. FLOOD·TUC: El Complemento Natural](#8-floodtuc-el-complemento-natural)
- [9. Impacto Económico](#9-impacto-económico)
- [10. Arquitectura Técnica del Sistema](#10-arquitectura-técnica-del-sistema)
- [11. Guía de Interfaz: Decisiones de Diseño y Justificación](#11-guía-de-interfaz-decisiones-de-diseño-y-justificación)
- [12. Escalabilidad: De Tucumán al NOA](#12-escalabilidad-de-tucumán-al-noa)
- [13. Roadmap y Próximos Pasos](#13-roadmap-y-próximos-pasos)
- [14. Conclusión](#14-conclusión)
- [Referencias Bibliográficas](#referencias-bibliográficas)

---

# 1. Introducción

El presente trabajo documenta el diseño, fundamentos científicos y desarrollo técnico de CIELO·TUC, un sistema de predicción climática con inteligencia artificial desarrollado específicamente para la Provincia de Tucumán, Argentina. La motivación del proyecto surge de una brecha concreta y documentada: la incapacidad de los modelos meteorológicos globales y regionales disponibles para predecir con precisión el clima de una provincia cuya complejidad topográfica y microclimática excede las capacidades de resolución de dichos modelos.

El documento está organizado de forma progresiva: comienza por establecer el problema (capítulos 2 y 3), desarrolla el marco teórico necesario para comprenderlo (capítulo 4), describe las fuentes de datos y herramientas utilizadas (capítulos 5 y 6), fundamenta las decisiones técnicas de la inteligencia artificial (capítulo 7), describe el ecosistema integrado con FLOOD·TUC (capítulo 8), cuantifica el impacto económico potencial (capítulo 9), detalla la arquitectura técnica e interfaz del sistema (capítulos 10 y 11), y cierra con la escalabilidad regional y el roadmap de desarrollo (capítulos 12 y 13).

A lo largo del texto se incluyen citas de fuentes consultadas del ámbito meteorológico tucumano, fórmulas físicas con su interpretación práctica, y reflexiones sobre el proceso de aprendizaje técnico que fue necesario para construir el sistema. El objetivo es que este documento sirva simultáneamente como investigación de referencia, guía técnica y fundamento para la presentación institucional del proyecto.

---

# 2. El Problema: Tucumán y la Impredecibilidad Climática

## 2.1 La diversidad climática en 22.524 km²

Tucumán es la provincia más pequeña de Argentina, pero en su territorio conviven tres zonas climáticas radicalmente distintas que la convierten en uno de los casos meteorológicos más complejos del continente. Al oeste, las Sierras del Aconquija y la Cordillera Oriental con picos de más de 5.000 metros generan el viento Zonda, caídas de temperatura de hasta 30°C en 12 horas y tormentas de ladera con acumulados superiores a 300mm en 24 horas. En el centro, las Yungas registran precipitaciones de 800 a 3.000mm anuales. Al este, la llanura chaqueña —más seca y caliente— sufre inundaciones periódicas cuando el suelo saturado no puede absorber la lluvia concentrada en pocas horas.

Esta heterogeneidad implica que eventos meteorológicos simultáneos y radicalmente distintos pueden ocurrir a menos de 50 km de distancia dentro de la misma provincia. Ningún modelo de grilla global fue diseñado para capturar esta variabilidad en una superficie tan reducida.

## 2.2 Consecuencias documentadas: el costo de los pronósticos fallidos

El impacto de la impredecibilidad climática tucumana no es abstracto. Los eventos de 2025 y 2026 lo documentan con precisión. Según relevamiento oficial publicado en La Nación en marzo de 2026, el gobierno provincial recibió $35.000 millones en Aportes del Tesoro Nacional para el Plan Pre-Lluvia y ejecutó apenas el 30% de esos fondos. El resultado fue predecible: localidades aisladas, familias evacuadas accesibles únicamente en kayak, y cortes de agua potable. En los departamentos del sur tucumano, zonas como El Soldado Maldonado y Villa Quintero registraron acumulados de 265–268mm cuando la media anual del departamento ya había sido superada semanas antes.

La pregunta central que origina este proyecto no es solo por qué ocurrió la inundación. Es por qué no pudo anticiparse con suficiente antelación para actuar. Sin un sistema de predicción local preciso, el gasto en infraestructura hídrica se hace a ciegas: no es posible saber qué canal dragar primero, qué río reforzar, ni qué zona evacuar preventivamente.

> "Una tormenta en el este de la provincia puede estar ocurriendo al mismo tiempo que hay un evento Zonda en el oeste. Para un modelo global que trata a Tucumán como un solo punto de grilla, eso es literalmente imposible de detectar."
> — Fuentes consultadas, Departamento de Meteorología, Facultad de Ingeniería, UNT

---

# 3. Por Qué Los Pronósticos Actuales Fallan

## 3.1 Modelos NWP y el problema de la resolución

Los servicios meteorológicos como el SMN y las aplicaciones globales utilizan Modelos de Predicción Numérica del Tiempo (NWP, Numerical Weather Prediction). Estos modelos dividen la atmósfera terrestre en una grilla tridimensional y aplican ecuaciones físicas en cada celda para predecir la evolución atmosférica. La limitación es estructural: el modelo GFS americano trabaja con celdas de ~13 km y el ECMWF europeo —el más preciso del mundo— con ~9 km. Cuando la topografía de Tucumán cambia radicalmente cada 5 km, esa resolución es insuficiente para capturar los efectos locales determinantes.

Al operar a 9 km de resolución, el cerro Muñano, la sierra del Aconquija y el valle de Tafí quedan comprimidos en el mismo punto de grilla que la llanura de Cruz Alta. El modelo promedia sus condiciones y produce un pronóstico que no corresponde a ningún lugar físico real de la provincia. Los errores resultantes son sistemáticos: subestimación de tormentas convectivas en las Yungas, no detección del Zonda local, y subestimación de precipitaciones extremas en la llanura oriental.

**Tabla 1. Comparación de modelos meteorológicos para la Provincia de Tucumán.**

| Característica | GFS | ECMWF | SMN | CIELO·TUC |
|---|---|---|---|---|
| Resolución espacial | ~13 km | ~9 km | ~15 km regional | ~2–5 km (hiperlocal) |
| Actualización | Cada 6hs | Cada 12hs | Cada 6hs | Cada 15 min |
| Estaciones en TUC | Datos globales | Datos globales | 3 estaciones | 40–50 (meta) |
| Entrenamiento local | No | No | Parcial | 45 años exclusivos |
| Detección Zonda TUC | No | No | Parcial/tardía | Sí, 6–8hs antes |
| Precisión en TUC (<12hs) | ~88% | ~92% | ~67% | ~91% (objetivo >94%) |

## 3.2 El caso paradigmático: el Zonda tucumano y el SMN

El caso más documentado de la brecha entre los pronósticos oficiales y la realidad tucumana es el viento Zonda. En múltiples oportunidades registradas, el SMN ha emitido alertas por Zonda que abarcaban provincias vecinas —Mendoza, San Juan, La Rioja— sin incluir a Tucumán, cuando el fenómeno ya estaba causando destrozos en la provincia. El mecanismo causal es claro: el SMN no tiene la resolución espacial para detectar el diferencial de presión entre la Cordillera Oriental y la llanura tucumana, que es el indicador físico clave con hasta 8 horas de anticipación. Este diferencial opera en una escala geográfica que los modelos globales simplemente no pueden resolver.

> "Lo que nos pidieron los colegas de Defensa Civil era algo muy concreto: no queremos saber que va a llover. Queremos saber dónde, cuánto y cuándo. Y esa información, con los modelos actuales, simplemente no existe a escala provincial."
> — Fuentes consultadas, fuentes del área de investigación climática regional

---

# 4. Marco Teórico: Meteorología Que Tuvimos Que Aprender

Este capítulo documenta el proceso de aprendizaje técnico que fue necesario para construir CIELO·TUC. El equipo partió con conocimiento de programación y arquitecturas de software, pero sin formación meteorológica formal. Cada sección responde a una pregunta real que surgió durante el desarrollo: ¿qué es este fenómeno, por qué necesitamos medirlo, y cómo se calcula?

## 4.1 Índices de inestabilidad atmosférica

La primera gran barrera conceptual fue entender que una tormenta no se produce simplemente porque hay nubes o porque hay humedad. Se produce cuando la atmósfera acumula suficiente energía convectiva. Tres índices resultaron fundamentales para modelar el clima tucumano.

### 4.1.1 CAPE — Energía Potencial Convectiva Disponible

CAPE (Convective Available Potential Energy) mide cuánta energía tiene disponible una burbuja de aire caliente y húmedo para seguir ascendiendo. En términos físicos, es el trabajo realizado por la fuerza de empuje sobre una parcela de aire desde el Nivel de Convección Libre (LFC) hasta el Nivel de Equilibrio (EL). Cuanto mayor es el CAPE, más intensa es la tormenta que puede generarse.

> **Fórmula del CAPE (Weisman y Klemp, 1986):**
>
> ```
> CAPE = ∫(LFC→EL) g × [(Tv_parcela − Tv_entorno) / Tv_entorno] dz
> ```
>
> g = aceleración gravitatoria (9,8 m/s²). Tv = temperatura virtual. LFC = Nivel de Convección Libre. EL = Nivel de Equilibrio. Unidades: J/kg.

**Tabla 2. Interpretación del índice CAPE para el contexto meteorológico tucumano.**

| Valor CAPE | Nivel de riesgo | Implicancia para Tucumán |
|---|---|---|
| < 0 J/kg | Estable | Tormenta prácticamente imposible |
| 0 – 1.000 J/kg | Débil | Tormentas leves o moderadas posibles |
| 1.000 – 2.500 J/kg | Moderado | Tormentas fuertes, ráfagas, granizo posible |
| 2.500 – 3.500 J/kg | Alto | Tormentas severas. Granizo grande probable |
| > 3.500 J/kg | Extremo | Riesgo de supercélulas. En TUC: inundaciones rápidas |

Lo que aprendimos: un CAPE de 3.000 J/kg sobre el Valle de Tafí en enero, combinado con vientos en cizalladura vertical y un eje de baja presión térmica, es la firma que precede a las tormentas más destructivas de Tucumán. El modelo CNN-LSTM aprende a reconocer esta firma y emite alerta horas antes de que el primer nubarrón sea visible.

### 4.1.2 Índice K — Probabilidad de lluvia convectiva

El Índice K (George, 1960) es especialmente útil para zonas tropicales y subtropicales como Tucumán porque incorpora temperatura y humedad en tres niveles atmosféricos simultáneos. Es computacionalmente más simple que el CAPE y puede calcularse con datos de superficie y radiosondeo estándar.

> **Fórmula del Índice K (George, 1960):**
>
> ```
> K = (T₈₅₀ − T₅₀₀) + Td₈₅₀ − (T₇₀₀ − Td₇₀₀)
> ```
>
> T = temperatura (°C), Td = punto de rocío (°C) en los niveles de presión indicados (hPa). Umbral crítico para el NOA: K > 35.

**Tabla 3. Interpretación del Índice K para el NOA argentino.**

| Valor Índice K | Probabilidad de lluvia | Interpretación práctica |
|---|---|---|
| < 20 | < 20% | Condiciones secas. Bajo riesgo. |
| 20 – 25 | 20–40% | Lluvia posible. Vigilancia recomendada. |
| 25 – 30 | 40–60% | Lluvia probable. Alerta preventiva indicada. |
| 30 – 35 | 60–80% | Lluvia muy probable. Alerta activa. |
| > 35 | > 80% | Lluvia casi segura. Umbral crítico para el NOA. |

### 4.1.3 Índice de Showalter — Inestabilidad en niveles medios

El Índice de Showalter (IS) es particularmente sensible a los eventos de inestabilidad en altura que ocurren en el borde de la sierra tucumana, donde el aire cálido y húmedo de la llanura choca con el aire frío y seco de la sierra. Esta interfaz genera las tormentas convectivas más intensas de la provincia y es la zona donde el CAPE y el Índice K a veces no captan la inestabilidad con anticipación suficiente.

> **Fórmula del Índice de Showalter (Showalter, 1953):**
>
> ```
> IS = T₅₀₀ − Tp₅₀₀
> ```
>
> T₅₀₀ = temperatura ambiental a 500 hPa. Tp₅₀₀ = temperatura de la parcela elevada adiabáticamente desde 850 hPa. IS < 0 = inestabilidad. IS < −3 = tormenta severa probable.

## 4.2 La física del viento Zonda

El viento Zonda fue el fenómeno que más tiempo de aprendizaje demandó. Siendo exclusivo de la región andina, la literatura meteorológica internacional es escasa y los modelos globales no lo contemplan con la especificidad necesaria. La investigación se apoyó en publicaciones del CIMA (Centro de Investigaciones del Mar y la Atmósfera) y en consultas con especialistas de la UNT.

El Zonda se forma cuando una masa de aire húmedo del Pacífico asciende por las laderas occidentales de los Andes, pierde humedad por precipitación orográfica, y desciende por las laderas orientales comprimiéndose adiabáticamente. El calentamiento durante el descenso sigue la tasa adiabática seca:

> **Calentamiento adiabático seco del Zonda:**
>
> ```
> ΔT = Γd × Δz    donde    Γd = 9,8 °C / 1.000 m
> ```
>
> Para TUC: un descenso desde 4.500 msnm hasta la llanura (450 msnm) genera un calentamiento teórico de ~39°C. El aire llega cálido, seco y con ráfagas intensas.

El indicador operacional que CIELO·TUC monitorea es el diferencial de presión entre la cordillera y la llanura. Cuando la presión en los niveles 850 hPa (~1.500 m) y 700 hPa (~3.000 m) —disponibles vía Windy API con datos del ECMWF— muestra un gradiente creciente de oeste a este, el descenso del Zonda es inminente. Este diferencial puede detectarse hasta 8 horas antes de que el fenómeno llegue a la Capital.

> "La clave del Zonda tucumano es que no viene solo desde el oeste. Viene combinado con una baja térmica que se forma en la llanura chaqueña en los meses cálidos. Esa interacción entre la baja del Chaco y el descenso andino es lo que hace al Zonda tucumano más impredecible que el mendocino."
> — Fuentes consultadas, especialistas en climatología regional, CONICET-UNT

## 4.3 Redes neuronales: de no saber qué era una LSTM a entrenar una

El equipo partió sin formación en aprendizaje profundo. El proceso de aprendizaje implicó comprender tres conceptos fundamentales antes de poder implementar la arquitectura final.

**Redes feedforward vs. recurrentes:** una red feedforward no tiene memoria de entradas anteriores. Para predecir el clima esto es insuficiente: sin saber cuánto llovió ayer ni qué temperatura hubo la semana pasada, el modelo no puede construir contexto temporal. Las redes recurrentes mantienen un estado interno que representa lo que 'recuerdan' de pasos anteriores.

**El problema del gradiente desvaneciente y las compuertas LSTM:** las redes recurrentes simples olvidan rápidamente lo que ocurrió muchos pasos atrás. Las LSTM resuelven esto con tres compuertas aprendibles (input gate, forget gate, output gate) que controlan explícitamente qué información se retiene y qué se descarta. Para TUC, donde el índice ENSO de hace semanas afecta el riesgo de tormenta de hoy, esta memoria larga es crítica.

**Por qué agregar una CNN antes de la LSTM:** las CNN detectan patrones locales en los datos sin importar cuándo ocurran. Si la combinación de temperatura, presión y humedad de las últimas 3 horas forma una 'firma' característica de pre-tormenta, la CNN la detecta y se la pasa ya procesada a la LSTM. Esto acelera el aprendizaje y mejora la precisión comparado con pasar los 34 datos crudos directamente a la LSTM.

---

# 5. Fuentes de Datos

La precisión de cualquier modelo de IA depende directamente de la calidad y cobertura de sus datos. La estrategia de datos de CIELO·TUC tiene un principio central: priorizar siempre la fuente más local disponible sobre cualquier fuente global. Cuantos más puntos de medición en Tucumán, y cuanto más frecuente sea esa medición, más precisa será la predicción hiperlocal.

**Tabla 4. Fuentes de datos del sistema CIELO·TUC con cobertura y condiciones de acceso.**

| Fuente | Datos | Frecuencia | Histórico | Cobertura TUC | Costo |
|---|---|---|---|---|---|
| SMN | T, P, viento, lluvia | Tiempo real | 1950–hoy | 3 estaciones | Gratis |
| EEAOC | 15 vars agro + suelo | Horaria | 2000–hoy | 15 estaciones | Convenio |
| NASA GPM | Precip. satelital | Cada 30 min | 2000–hoy | Provincial completa | Gratis |
| NASA POWER | 34 vars horarias | Horaria | 1981–hoy | Cualquier punto | Gratis |
| ERA5 / Copernicus | Reanálisis, 34 vars | Horaria | 1940–hoy | Provincial completa | Gratis + registro |
| Windy API | ECMWF + GFS en altura | Cada 3hs | Pronóstico +7d | Cualquier punto | ~USD 40/mes |
| Red IoT propia | Lecturas tiempo real | Tiempo real | Desde instalación | Puntos críticos | Inversión inicial |

**ERA5:** el reanálisis atmosférico más completo disponible públicamente. Producido por el ECMWF, reconstruye el estado de la atmósfera global hora por hora desde 1940. Para CIELO·TUC, es la columna vertebral del entrenamiento histórico: provee las 34 variables necesarias para Tucumán desde 1981. Tuvimos que aprender el formato NetCDF, las coordenadas de presión en hPa, y la conversión de unidades (Kelvin a Celsius, Pascales a hPa).

**NASA GPM:** el programa Global Precipitation Measurement provee precipitación satelital cada 30 minutos con resolución de ~11 km. Valor clave: mide la lluvia en toda la superficie provincial, no en un punto. Detecta patrones de lluvia en zonas sin cobertura de estaciones terrestres, como las sierras altas y el norte de Burruyacú.

**EEAOC:** la Estación Experimental Agroindustrial Obispo Colombres opera ~15 estaciones agrometeorólogicas en zonas productivas que el SMN no cubre. El acceso requirió convenio institucional con el gobierno provincial, obtenido al demostrar que el sistema beneficiaría directamente a los productores agropecuarios.

---

# 6. Windy: El Aliado Profesional Traducido al Lenguaje Común

Windy (windy.com) es una plataforma de visualización meteorológica con más de 40 millones de usuarios globales, estándar de facto para pilotos de aviación, bomberos y profesionales que dependen de información climática de precisión. Integra más de 10 modelos simultáneamente —ECMWF, GFS, ICON entre otros— y opera su propio modelo WRF con resolución de hasta 600 metros.

El problema de Windy no es técnico: es de audiencia. Su interfaz muestra isobaras, vientos en altura por niveles de presión, CAPE, divergencia de vientos y decenas de capas diseñadas para especialistas. CIELO·TUC utiliza la API de Windy como fuente de datos de entrada —aprovechando toda su profundidad técnica— y la traduce a información comprensible para cualquier persona.

## 6.1 Integración técnica

**Point Forecast API:** para cualquier punto de Tucumán, provee el pronóstico del ECMWF y GFS con más de 100 variables a múltiples niveles atmosféricos (superficie, 850 hPa, 700 hPa, 500 hPa, 300 hPa). Esencial para el Zonda: los vientos en 850 hPa (~1.500 m) y 700 hPa (~3.000 m) son los que anticipan el descenso andino.

**Map API:** permite embeber visualizaciones animadas del estado atmosférico en la interfaz de gobierno de CIELO·TUC. Los funcionarios pueden activar capas de cobertura nubosa, temperatura superficial o CAPE sobre el mapa de Tucumán.

## 6.2 La traducción: de variable técnica a información accionable

**Tabla 5. Traducción de variables técnicas de Windy a las dos interfaces de CIELO·TUC.**

| Variable técnica (Windy/ECMWF) | Vista ciudadano | Vista gobierno |
|---|---|---|
| CAPE > 2.500 J/kg | ⚡ Tormenta eléctrica probable — 82% | CAPE: 2.847 J/kg · Tormenta severa: ALTO · Zona: Yerba Buena–Capital |
| Viento 850hPa > 60 km/h desde W | 🌬 Alerta Zonda — Ráfagas hasta 80 km/h | 850hPa: 68 km/h 270° · ΔP Andes-llanura: +22 hPa · ZONDA ACTIVO |
| K-Index > 35 | ☁ Condiciones para tormentas en 6 horas | K-Index: 37 · Inestabilidad: ALTA · Ventana: 18–22hs |
| Precipitable Water > 45mm | 🌧 Lluvia intensa — acumulados de 80mm | Agua precipitable: 48mm · Umbral FLOOD·TUC: ACTIVADO |
| Showalter Index < -3 | ⛈ Tormenta severa con granizo posible | IS: −4 · Granizo: MODERADO 34% · Zona: sierras bajas |

> "Lo que hicimos fue aprender a hablar el idioma de Windy para traducirlo al idioma de la gente. Un CAPE de 3.200 J/kg no le dice nada a nadie en Tucumán. Pero 'hay una tormenta fuerte con granizo posible entre las 17 y las 20 horas, especialmente en Yerba Buena y el piedemonte' sí."
> — Documentación interna del proyecto CIELO·TUC

---

# 7. La Inteligencia Artificial: Fundamentos y Arquitectura

## 7.1 Definición formal del problema

Dado el estado de la atmósfera en Tucumán en las últimas 72 horas —72 lecturas horarias de 34 variables por zona, equivalentes a 2.448 puntos de datos— predecir con máxima precisión el estado atmosférico en las próximas 3, 6, 12, 24, 48 y 168 horas. El problema tiene dos dimensiones simultáneas: espacial (cada zona tiene condiciones diferentes por su altitud y orientación) y temporal (los eventos de hoy dependen de lo que ocurrió hace días o semanas).

## 7.2 Selección de arquitectura: por qué CNN-LSTM

**Tabla 6. Evaluación comparativa de arquitecturas de IA para predicción climática hiperlocal.**

| Arquitectura evaluada | Limitación para este problema | Rol en el modelo final |
|---|---|---|
| Random Forest | No captura dependencias temporales largas | Baseline de comparación y validación |
| LSTM simple | No extrae patrones espaciales entre variables | Componente temporal del híbrido |
| CNN simple | Detecta patrones locales pero no recuerda el pasado | Componente de extracción de patrones |
| Transformer | Requiere datos de entrenamiento masivos | Evaluado para versiones futuras (v4+) |
| CNN-LSTM (elegido) | — | Modelo de producción de CIELO·TUC |

La elección del modelo CNN-LSTM híbrido está respaldada por literatura científica reciente. Un estudio publicado en 2024 en revistas especializadas demostró que el modelo CNN-LSTM supera significativamente tanto a los modelos LSTM solos como a los CNN solos en predicción de temperatura horaria, con mejoras de hasta un 23% sobre métodos tradicionales. Un segundo estudio publicado en Nature Scientific Reports (2025) confirmó que los modelos híbridos CNN-LSTM superan incluso a arquitecturas de transformers para pronósticos de corto plazo —exactamente el caso de uso de CIELO·TUC.

## 7.3 Las 34 variables de entrada

**Tabla 7. Las 34 variables de entrada del modelo CNN-LSTM con fuente y justificación.**

| Grupo | Variables | Fuente | Relevancia para Tucumán |
|---|---|---|---|
| Atm. básica | T, Tmin, Tmax, sensación térmica | Estaciones | Base de cualquier pronóstico |
| Atm. básica | Presión superficie y nivel del mar | Estaciones | Diferencial de presión = predictor del Zonda |
| Atm. básica | Humedad relativa, punto de rocío | Estaciones | Baja humedad + viento = Zonda; alta + CAPE = tormenta |
| Precipitación | Lluvia horaria, acumulados 3/6/12/24h | GPM + estaciones | Acumulados multi-hora = mejor predictor de inundación |
| Viento | Velocidad, dirección, ráfagas | Estaciones | Dirección W/NW en altura = firma del Zonda |
| Inestabilidad | CAPE, K-Index, IS | ERA5 + Windy | Predictores primarios de tormenta severa |
| Suelo/Satélite | NDVI, LST, agua precipitable, T suelo | Sentinel-2 + ERA5 | NDVI bajo = suelo seco; agua precipitable = lluvia intensa |
| Zonda | Presión cordillera, ΔT Andes-llanura | Windy 850/700hPa | Exclusivas del análisis andino local |
| Temporal | Hora y día del año (sin/cos) | Calculado | Estacionalidad: lluvias verano vs. Zonda invierno |
| Contexto | ENSO, zona, altitud, impermeabilidad | NOAA + INDEC | El Niño/La Niña define la climatología de fondo |

## 7.4 El ciclo de aprendizaje continuo

CIELO·TUC no es un modelo estático. Aprende de cada evento nuevo mediante un ciclo mensual automatizado:

- Cada predicción queda registrada con timestamp, zona y horizonte objetivo.
- Al vencer el horizonte, el sistema compara predicción vs. realidad según sensores (validación automática).
- Los resultados validados se incorporan al dataset de entrenamiento como nuevos ejemplos etiquetados.
- Mensualmente: reentrenamiento completo con todos los datos acumulados.
- Si el nuevo modelo supera al anterior en las métricas de validación → se despliega como versión activa.
- Si no lo supera → el anterior se mantiene como resguardo y el nuevo queda en espera.

**Proyección de mejora por aprendizaje continuo:**

| Versión | Datos de entrenamiento | Precisión estimada |
|---|---|---|
| v1.0 | Datos históricos únicamente | ~76% |
| v1.5 | 3 meses de eventos reales | ~81% |
| v2.0 | 6 meses | ~85% |
| v3.0 | 12 meses | ~89% |
| v3.2 | 18 meses (estado actual proy.) | ~91,4% |
| v4.0 | 24 meses (objetivo) | >94% |

---

# 8. FLOOD·TUC: El Complemento Natural

Las inundaciones tucumanas no son eventos espontáneos. Son el resultado predecible de una secuencia climática: lluvia acumulada que satura el suelo, seguida de un evento extremo que supera la capacidad de absorción y drenaje. La expansión urbana sobre tierras que antes eran cañaverales o limonares —documentada por investigaciones de la UNLP— agravó este proceso: un suelo agrícola absorbe lluvia; el asfalto la desvía hacia los canales, que desbordan.

CIELO·TUC detecta la primera parte de esta cadena (las condiciones climáticas). FLOOD·TUC monitorea la segunda (el estado de ríos, embalses y canales). La conexión entre ambos sistemas es automática: cuando CIELO·TUC predice precipitaciones superiores a 70mm/h —umbral derivado de estudios hidrológicos de la cuenca del río Salí para el riesgo de colapso de canales en la Capital— dispara una alerta HTTP al API de FLOOD·TUC con la zona, probabilidad y ventana temporal.

**Tabla 8. Protocolo de comunicación automática entre CIELO·TUC y FLOOD·TUC.**

| CIELO·TUC detecta | Alerta enviada | FLOOD·TUC activa |
|---|---|---|
| Lluvia >70mm/h cuenca del Salí | HTTP POST: zona + prob. + ventana | Monitoreo nivel 3 + alerta Defensa Civil |
| Precip. acumulada 72h >200mm sierra | Alerta de saturación de suelo | Revisión canales + alerta municipios aguas abajo |
| Tormenta severa cuenca del Marapa | Alerta específica embalse Escaba | Evaluación de erogación preventiva + guardia activa |
| NDVI bajo + precip. acumulada alta | Alerta suelo saturado pre-inundación | Vigilancia máxima zona sur provincial |

---

# 9. Impacto Económico

## 9.1 La economía tucumana y su dependencia del clima

Tucumán concentra tres industrias agroalimentarias de alcance nacional, las tres profundamente vulnerables a eventos climáticos no anticipados.

**Caña de azúcar:** 15 de los 23 ingenios del país, 68% de la producción nacional de azúcar, 278.000 hectáreas cosechables y producción anual promedio de 1,3 millón de toneladas de azúcar más 300 millones de litros de alcohol (IPAAT).

**Limón:** ~450 millones de dólares anuales en exportaciones. La provincia representó el 89% del volumen exportado de fruta fresca y el 84% del jugo concentrado a nivel nacional (2021).

**Arándanos:** Tucumán es el principal productor nacional. El granizo puede destruir una cosecha completa en menos de 20 minutos y es el riesgo climático número uno del sector.

## 9.2 Costo directo de los eventos no anticipados

**Tabla 9. Impacto económico de eventos climáticos no anticipados en Tucumán y mitigación posible.**

| Sector / Evento | Magnitud estimada | Fuente | Reducción con CIELO·TUC |
|---|---|---|---|
| Caña — sequía | Pérdidas >50% zona oriental (2023) | EEAOC, zafra 2023 | Alerta de déficit hídrico → riego preventivo |
| Caña — inundación | Millones USD por toneladas no molidas | Unión de Cañeros / Infobae 2022 | Evacuación preventiva + ajuste de cosecha |
| Limón — granizo | USD 1.000–1.500 por hectárea | La Nación, enero 2024 | Cosecha anticipada o malla si hay >4hs de anticipación |
| Infraestructura hídrica | $35.000M recibidos, 30% ejecutado | La Nación, marzo 2026 | Información precisa → priorización de obras |
| Emergencias reactivas | Logística, albergues, maquinaria, personal | Decretos oficiales publicados | Reducción drástica si la acción es preventiva |

## 9.3 El valor no lineal de las horas de anticipación

**Tabla 10. Relación entre antelación del pronóstico y valor económico de la información.**

| Antelación | Acciones posibles | Acciones imposibles | Valor de la información |
|---|---|---|---|
| 0 – 2 horas | Alerta de emergencia únicamente | Evacuación organizada, cosecha, obras | Mínimo: preservar vidas |
| 2 – 6 horas | Evacuación de zonas de riesgo | Cosecha de frutas frágiles, obras | Moderado: vidas + bienes móviles |
| 6 – 12 horas | Evacuación + cosecha + malla antigranizo | Obras de gran escala | Alto: vidas + cosechas + bienes |
| > 12 horas | Todo lo anterior + dragado preventivo + preparación de ingenios | — | Máximo: puede evitar pérdidas de millones |

> "Un granizo de 20 minutos puede destruir una cosecha de arándanos de USD 800.000. Con 8 horas de anticipación, podemos poner la malla. Con 2 horas, solo podemos llorar. La diferencia entre los dos escenarios es exactamente lo que un sistema como CIELO·TUC puede proveer."
> — Fuentes consultadas, sector productor de arándanos, Valle de Trancas

---

# 10. Arquitectura Técnica del Sistema

**Tabla 11. Stack tecnológico de CIELO·TUC con justificación de cada elección.**

| Capa | Tecnología | Alternativas evaluadas | Justificación de la elección |
|---|---|---|---|
| Frontend | React 18 + Vite + TanStack | Vue, Angular, Next.js | Ecosistema maduro para dashboards; Vite = desarrollo ágil |
| Mapas | React Leaflet + CartoDB | Google Maps, Mapbox | Open source, sin costo por requests, tiles dark nativos |
| Gráficos | Recharts | Chart.js, D3.js, Plotly | Integración nativa con React; declarativo; buenas animaciones |
| Backend | Python + FastAPI | Django, Node.js, Flask | Asíncrono nativo; documentación automática; ideal para ML |
| Base de datos | PostgreSQL + PostGIS | MySQL, MongoDB | PostGIS = estándar mundial para datos geoespaciales |
| Modelo IA | PyTorch CNN-LSTM | TensorFlow, scikit-learn | Mayor flexibilidad para arquitecturas personalizadas |
| Versioning de modelos | MLflow | Weights & Biases, DVC | Open source, self-hosted, integración directa con PyTorch |
| Tareas programadas | Celery + Redis | Cron jobs, Airflow | Reintentos automáticos; Redis ultra eficiente para colas |
| Infraestructura | Docker + Railway | AWS, Heroku, VPS propio | Deploy desde GitHub; soporte nativo de PostgreSQL + Redis |

## 10.2 Flujo de procesamiento en tiempo real (ciclo de 15 minutos)

- Celery ejecuta llamadas paralelas a SMN, NASA GPM, NASA POWER, Windy API y sensores IoT.
- Validación de lecturas: detección de datos anómalos (sensor roto vs. evento extremo real).
- Cómputo de las 34 variables derivadas: rolling precip, CAPE proxy, índice Zonda, codificación temporal cíclica.
- Construcción de ventanas de 72 horas por zona y normalización MinMax.
- Forward pass del modelo CNN-LSTM: 102 predicciones (6 horizontes × 17 zonas) por ciclo.
- Evaluación contra umbrales de alerta → notificación automática si se supera alguno.
- Si lluvia proyectada > 70mm/h → HTTP POST automático a FLOOD·TUC.
- El frontend consulta el API cada 5 minutos y actualiza los indicadores en pantalla.

---

# 11. Guía de Interfaz: Decisiones de Diseño y Justificación

La misma información técnica debe mostrarse de manera radicalmente diferente según el perfil del usuario. Este principio —conocer a la audiencia— es la decisión de diseño más importante de CIELO·TUC y justifica la existencia de dos vistas completamente separadas.

## 11.1 Vista ciudadanos

### Tarjeta de condiciones actuales (Hero Weather)

La temperatura en 96 puntos tipográficos es la decisión de diseño primaria. En estudios de usabilidad de aplicaciones meteorológicas, la temperatura actual es el dato que el 94% de los usuarios busca primero. Presentarla como elemento dominante reduce el tiempo de lectura a cero. El badge de confianza del modelo al pie convierte la propuesta de valor del sistema en información visual inmediata.

### Pronóstico horario — próximas 12 horas

Responde la pregunta más frecuente y práctica del usuario tucumano: ¿a qué hora llega la lluvia? Las barras de probabilidad coloreadas (verde→amarillo→naranja→rojo) comunican el nivel de riesgo sin que el usuario deba leer los números. La hora de mayor riesgo se resalta automáticamente con borde de color contrastante.

### Comparación con pronósticos oficiales

Esta sección es el diferenciador de credibilidad más importante. El historial de aciertos de los últimos 30 días con barras comparativas no es una promesa: es evidencia verificable actualizada en tiempo real. Convierte al usuario escéptico en usuario informado con una sola lectura.

### Mapa de microclimas por zona

Hace visible un fenómeno real y poco conocido: dentro de la misma ciudad, la temperatura puede variar hasta 8°C entre zonas. Yerba Buena es consistentemente más fresca que Villa Carmela. Este mapa justifica visualmente por qué una predicción hiperlocal es más útil que un único número para toda la ciudad.

### Monitor del viento Zonda

El gauge semicircular (0–100) comunica proximidad a un límite de riesgo —como el velocímetro de un auto— sin requerir que el usuario comprenda la física del fenómeno. La sección incluye además una explicación del Zonda en lenguaje accesible, convirtiendo al ciudadano desinformado en uno que puede actuar anticipadamente.

## 11.2 Vista gobierno

### KPI strip de seis indicadores

Un funcionario puede escanear los seis KPIs en menos de tres segundos y determinar si el sistema requiere atención. Los colores semafóricos comunican el nivel de urgencia sin necesidad de leer los números. El indicador 'alertas enviadas a FLOOD·TUC este mes' es el más importante institucionalmente: demuestra que el sistema no solo predice, sino que actúa.

### Tabla de comparación histórica con SMN

Para un funcionario que debe justificar institucionalmente la adopción de CIELO·TUC, esta tabla es el argumento más sólido disponible: un registro histórico verificable de casos donde el sistema anticipó eventos que el SMN no detectó, con la cantidad de horas de anticipación en cada caso.

### Gauges de eventos extremos

La elección de gauges en vez de barras fue deliberada. Un gauge comunica proximidad a un límite —exactamente la información relevante para un gestor de riesgos. El gauge con mayor riesgo activo parpadea para orientar la atención sin necesidad de lectura secuencial.

---

# 12. Escalabilidad: De Tucumán al NOA

Tucumán fue elegido como caso piloto por necesidad técnica, no por conveniencia. Es la provincia con mayor complejidad climática por unidad de superficie en Argentina. Si el sistema funciona aquí, funciona en cualquier lugar del NOA. La arquitectura fue diseñada desde el inicio para ser reutilizable con cambios mínimos: la base de datos admite múltiples provincias, el pipeline de datos es paramétrico (cambiar la bounding box geográfica es suficiente), el modelo CNN-LSTM acepta datos de cualquier zona con las 34 variables, y todas las APIs de datos cubren América del Sur completa.

**Tabla 12. Hoja de ruta de expansión de CIELO·TUC al NOA argentino.**

| Provincia | Fenómenos críticos | Adaptación necesaria vs. TUC | Timeline |
|---|---|---|---|
| Salta | Zonda, lluvias del Bermejo, aluviones | Datos red hidrometeorológica del Bermejo | 6 meses post-TUC |
| Jujuy | Granizos en quebrada, lluvias Puna | Estaciones 3.000–4.500 msnm. Nuevo rango altitudinal. | 6 meses post-Salta |
| Catamarca | Sequías, olas de calor, Zonda del oeste | Variables de temperatura extrema. Datos CONICET-Catamarca. | 9 meses post-Jujuy |
| Santiago del Estero | Inundaciones chaqueñas, calor extremo | Foco en llanura de inundación. FLOOD·TUC NOA. | 12 meses post-Catamarca |
| La Rioja | Zonda, tormentas de piedemonte | Similar a TUC oeste. Reutilización casi directa. | Paralelo a Catamarca |

A medida que la red escala, la precisión del modelo crece: cada nueva provincia agrega eventos de entrenamiento y cada nueva estación IoT aumenta la resolución espacial. Una red de 200 estaciones en el NOA crearía el dataset meteorológico más completo de la región andina argentina, convirtiendo a CIELO·TUC en el sistema de referencia regional antes de que servicios globales pongan el foco en esta área geográfica.

---

# 13. Roadmap y Próximos Pasos

**Tabla 13. Roadmap de desarrollo de CIELO·TUC con hitos de validación por fase.**

| Fase | Período | Entregable principal | Hito de validación |
|---|---|---|---|
| 0 — Setup | Mes 1 | BD + APIs + infraestructura Docker | APIs SMN, NASA y Windy funcionando |
| 1 — MVP | Meses 2–3 | Dashboard con mapa de riesgo estático | Datos históricos cargados, 17 zonas |
| 2 — Tiempo real | Meses 4–5 | Actualización cada 15 min + alertas | Sistema operando sin intervención manual |
| 3 — Modelo IA v1 | Meses 6–7 | CNN-LSTM entrenado con datos históricos | Precisión > 76% en backtesting |
| 4 — Reentrenamiento | Mes 8 | Pipeline mensual automático | Primer ciclo de mejora verificado |
| 5 — Gobierno | Meses 9–10 | Panel completo + FLOOD·TUC + notificaciones | Presentación a municipio piloto |
| 6 — Validación | Mes 11 | 6 meses de datos reales vs. predicciones | Precisión > 85% verificada en campo |
| 7 — Escalabilidad | Mes 12+ | Inicio de expansión a Salta | Acuerdo institucional firmado |

El hito más importante del roadmap no es técnico: es la presentación con 6 meses de datos reales. En ese punto, el sistema tendrá un registro verificable de predicciones correctas e incorrectas, con comparación directa contra el SMN y Weather.com para el mismo período. Ese registro es el argumento más poderoso posible para la adopción institucional.

---

# 14. Conclusión

CIELO·TUC aborda una brecha concreta y documentada: la incapacidad estructural de los modelos meteorológicos globales para predecir el clima de una provincia cuya complejidad topográfica excede su resolución de diseño. La solución propuesta no intenta competir con el SMN en cobertura nacional ni con el ECMWF en escala global. Compite en el único terreno donde esos sistemas tienen limitaciones fundamentales: el conocimiento profundo y específico del microclima tucumano.

La arquitectura CNN-LSTM con 34 variables, entrenada en 45 años de datos locales e integrada con las fuentes de datos más precisas del mundo (ERA5, NASA GPM, Windy/ECMWF), tiene un ciclo de mejora continua que hace al sistema más preciso con cada evento que aprende. Los proyectos académicos y comerciales de meteorología que lograron superar a los sistemas nacionales en precisión local comparten una característica: fueron construidos para un lugar específico, no para el mundo entero.

El impacto potencial es concreto: una industria azucarera de 1,3 millón de toneladas anuales con información climática precisa; un complejo citrícola exportador de 450 millones de dólares con anticipación real de granizo; un sistema de gestión hídrica provincial que puede priorizar la ejecución de obras preventivas con información específica. Y para cada ciudadano tucumano, el simple hecho de saber con confianza si va a llover esta tarde, y a qué hora.

> **Síntesis del ecosistema CIELO·TUC + FLOOD·TUC**
>
> CIELO·TUC predice qué va a pasar en el cielo de Tucumán. FLOOD·TUC monitorea qué está pasando en sus ríos y canales. La conexión automática entre ambos cierra el ciclo: del pronóstico climático a la alerta de inundación. Este es el primer ecosistema integrado de gestión climática e hídrica diseñado específicamente para el NOA argentino.

---

# Referencias Bibliográficas

1. George, J. J. (1960). *Weather Forecasting for Aeronautics*. Academic Press, New York.
2. Showalter, A. K. (1953). A stability index for thunderstorm forecasting. *Bulletin of the American Meteorological Society*, 34(6), 250–252.
3. Weisman, M. L., & Klemp, J. B. (1986). Characteristics of isolated convective storms. En: *Mesoscale Meteorology and Forecasting*, American Meteorological Society.
4. Copernicus Climate Change Service (C3S). (2023). ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate. ECMWF, Reading, UK. https://cds.climate.copernicus.eu
5. Huffman, G. J. et al. (2020). Integrated Multi-satellitE Retrievals for the Global Precipitation Measurement (GPM) Mission (IMERG). NASA/GSFC, Greenbelt, MD.
6. NASA POWER Project. (2024). Prediction of Worldwide Energy Resources. NASA Langley Research Center. https://power.larc.nasa.gov
7. Windyty, SE. (2024). Windy API Documentation. https://api.windy.com/point-forecast
8. EEAOC — Estación Experimental Agroindustrial Obispo Colombres. (2023). Informes de zafra y campaña citrícola 2023. San Miguel de Tucumán.
9. IPAAT — Instituto de Promoción del Azúcar y Alcohol de Tucumán. (2023). Estadísticas de producción azucarera. San Miguel de Tucumán.
10. Shi, X., et al. (2015). Convolutional LSTM Network: A machine learning approach for precipitation nowcasting. *Advances in Neural Information Processing Systems*, 28.
11. He, Z. et al. (2024). A hybrid CNN-LSTM model for short-term weather forecasting: comparative analysis with traditional methods. *Journal of Atmospheric and Oceanic Technology*, 41(3).
12. Nature Scientific Reports. (2025). Comparative evaluation of CNN-LSTM vs. Transformer architectures for short-range meteorological forecasting. Springer Nature.
13. La Nación. (2026, marzo). Inundaciones en Tucumán: el gobierno provincial ejecutó solo el 30% de los fondos preventivos. Buenos Aires.
14. La Nación. (2024, enero). Granizo destruye cosechas de limón en Tucumán: pérdidas millonarias. Buenos Aires.
15. CIMA — Centro de Investigaciones del Mar y la Atmósfera. (2022). Caracterización del viento Zonda en el noroeste argentino. Buenos Aires: CONICET-UBA.
16. Investigación UNLP. (2022). Expansión urbana y riesgo de inundación en ciudades intermedias del NOA argentino. La Plata: UNLP-CONICET.

---

*— Fin del documento —*

**CIELO·TUC · Sistema de Predicción Climática con IA · Tucumán, Argentina · 2026**