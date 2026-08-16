# Reglas Estables del Repositorio DeltaPhase

## Filosofía y Visión
**EL NORTE DEL PROYECTO:**
1. **Eficiencia Subcuadrática y Memoria O(N):** Resolver las limitaciones de la atención cuadrática ($O(N^2)$) y el *memory crosstalk* en espacios lineales reales mediante álgebras de fasores unitarios en el círculo complejo $S^1$ ($\mathbb{C}^{d_k \times d_k}$).
2. **Elegancia y Precisión Matemática:** Priorizar formulaciones continuas y estables (contracción $\beta \in (0, 2)$, $T_{\text{mat}}$ triangular solve exacto, quasi-ortogonalidad) sobre la fuerza bruta.
3. **Rigor Científico y Falsabilidad:** Todo claim de superioridad o capacidad debe respaldarse con protocolos estandarizados (Zoology MQAR, NIAH, ablations iso-paramétricas) y controles negativos/positivos explícitos.

---

## Modificación de Código y Experimentos
- **Aislamiento y Trazabilidad:** Todo nuevo experimento o benchmark debe crearse en archivos autocontenidos y versionados (por ejemplo en `tests/` o `examples/`).
- **Vectorización Obligatoria:** Queda terminantemente prohibido usar bucles `for` de Python en forward passes de operaciones tensoriales. Toda proyección, mezcla y solución triangular debe ejecutarse mediante multiplicaciones matriciales (`@`) o primitivas optimizadas de PyTorch/Triton.

---

## 🌟 Regla de Trazabilidad, Cabeceras y Feedback en Tiempo Real

Todo script de experimento, benchmark o entrenamiento debe implementar obligatoriamente el siguiente estándar de visibilidad y monitoreo:

### 1. Cabecera con Metadatos, Explicación e Inventario de Arquitectura
Al iniciarse, el script debe imprimir una cabecera clara y delimitada que contenga:
- **Título y Propósito:** Nombre del experimento y breve explicación conceptual de qué hipótesis se está evaluando, comparando o refutando.
- **Inventario de Ejecución:** Dispositivo utilizado (CPU / GPU / Torch DirectML), versión de Python y PyTorch, commit hash de Git y fecha/hora UTC.
- **Configuración Completa:** Hiperparámetros explícitos (dimensiones $d_{\text{model}}$, cabezas $n_{\text{heads}}$, tamaño de bloque chunk $C$, rango de gating $\beta$, semillas, longitudes $L$, número de pasos y optimizador).
- **Inventario Detallado de Arquitectura:** Para cada modelo a evaluar, imprimir:
  - Parámetros totales y entrenables (`total` y `trainable params`).
  - Dimensiones estructurales ($d_{\text{model}}$, $n_{\text{heads}}$, $d_k$, vocabulario, profundidad).
  - Desglose por capas y submódulos (nombre, clase y parámetros por componente).

### 2. Feedback Continuo en Tiempo Real con Marcas de Tiempo
- **Marcas Temporales:** Cada línea de log emitida debe comenzar con una marca de tiempo absoluta `[HH:MM:SS]` o relativa `[+HH:MM:SS.ss]`.
- **Salida sin Búfer (`flush=True`):** Todo `print()` debe incluir el argumento `flush=True` (o el script ejecutarse con `python -u`) para asegurar que el usuario vea la salida en streaming en la consola inmediatamente, sin retrasos por búfer de stdout.
- **Monitoreo de Progreso y Métricas Intermedias:** Registrar periódicamente:
  - Paso actual, total de pasos y porcentaje de avance (`[Paso 200/800 (25.0%)]`).
  - Pérdida de entrenamiento reciente (`Train Loss`).
  - Métrica de evaluación retenida (`Held-out Acc / Val Loss`).
  - Velocidad de cómputo en pasos por segundo (`st/s`).
- **Estimación de Tiempo Restante (ETA):** Calcular y mostrar tanto el tiempo estimado restante para el modelo actual (**ETA modelo**) como la estimación para la suite completa (**ETA total suite**), además del tiempo global transcurrido.

---

## Niveles de Rigor Experimental

### Nivel 1 — Sondeo Exploratorio
- 1 semilla es admisible para comprobaciones rápidas de hipótesis o sanity checks (`--quick`).
- Se etiqueta como `[SEÑAL]` o `[EXPLORATORIO]`. Prohibido citarlo como evidencia definitiva o superioridad concluyente.

### Nivel 2 — Hallazgo Certificado / ANCLA
- Obligatorio un mínimo de **3 a 5 semillas independientes**.
- Evaluación en conjuntos de datos dinámicos *on-the-fly* o splits de test rigurosamente retenidos (fuera de distribución OOD si aplica).
- Reportar **media ± desviación estándar / error estándar (SE)**.
- Solo entonces un resultado puede ser promovido a `[ANCLA]` y citado en la documentación formal y artículos del proyecto.

---

## Hardware y Entorno
- **GPU:** AMD Radeon 780M / DirectML (`torch_directml`) o procesador multinúcleo AMD Ryzen 7 8845HS.
- **Python / PyTorch:** Python 3.10+ con PyTorch 2.0+.
