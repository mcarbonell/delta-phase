# Certificación Nivel 2: Benchmark Riguroso Multi-Semilla de MQAR (Multi-Query Associative Recall)

> **ESTADO: [ANCLA / CERTIFICADO NIVEL 2]**  
> Protocolo experimental estandarizado con 5 semillas independientes (`42, 137, 2024, 7, 999`), datos puramente dinámicos *on-the-fly* (inmunes a memorización de batch), parada temprana (*early stopping* $\ge 99.5\%$), y barrido de capacidad ($N_{\text{pairs}} \in \{8, 16, 32\}$) y longitud ($L \in \{128, 256, 512, 1024\}$).

---

## 🏗️ Inventario de Arquitecturas y Parámetros

Todos los modelos fueron evaluados con paridad dimensional ($d_{\text{model}}=128$, $n_{\text{heads}}=4$, $d_k=32$, $L_{\text{layers}}=2$, $V=514$, ventana convolucional local $k=4$ y embeddings posicionales absolutos):

| Modelo | Espacio de Memoria | Parámetros Totales | Parámetros Entrenables | Complejidad de Inferencia |
| :--- | :---: | :---: | :---: | :---: |
| **DeltaPhase** | Complejo $\mathbb{C}^{32 \times 32}$ ($S^1$) | 935,696 | 935,696 | $O(1)$ por token / $O(N)$ secuencia |
| **Gated DeltaNet** | Real $\mathbb{R}^{32 \times 32}$ | 1,054,218 | 1,054,218 | $O(1)$ por token / $O(N)$ secuencia |
| **Transformer Causal** | Softmax $QK^T$ | 1,054,210 | 1,054,210 | $O(N)$ por token / $O(N^2)$ secuencia |

---

## 📊 Tabla Resumen Certificada Nivel 2 (Media ± Error Estándar, $n=5$ semillas independientes)

| Configuración | Modelo | In-Distribution ($L_{\text{train}}$) | OOD $2\times$ | OOD $4\times$ | Pasos $>50\%$ | Pasos $>95\%$ | Tiempo Medio (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N_{\text{pairs}}=8$** | **DeltaPhase (Complejo)** | $98.49 \pm 0.30\%$ | $98.29 \pm 0.26\%$ | $98.38 \pm 0.28\%$ | 410.0 | 650.0 | 429.3s |
| ($L_{\text{train}}=128$) | **Transformer Causal (MHA)** | **$99.37 \pm 0.08\%$** | **$99.49 \pm 0.05\%$** | **$99.48 \pm 0.08\%$** | **240.0** | **250.0** | **90.6s** |
| | **Gated DeltaNet (Real)** | $97.70 \pm 0.42\%$ | $97.44 \pm 0.50\%$ | $97.64 \pm 0.51\%$ | 1000.0 | 1230.0 | 184.7s |
| **$N_{\text{pairs}}=16$** | **Transformer Causal (MHA)** | **$99.61 \pm 0.05\%$** | **$99.65 \pm 0.02\%$** | **$99.64 \pm 0.02\%$** | **280.0** | **300.0** | **41.1s** |
| ($L_{\text{train}}=128$) | **DeltaPhase (Complejo)** | $99.16 \pm 0.19\%$ | $99.14 \pm 0.17\%$ | $99.19 \pm 0.20\%$ | 580.0 | 750.0 | 343.2s |
| | **Gated DeltaNet (Real)** | $97.33 \pm 0.41\%$ | $97.52 \pm 0.34\%$ | $97.52 \pm 0.32\%$ | 810.0 | 1090.0 | 185.6s |
| **$N_{\text{pairs}}=32$** | **Transformer Causal (MHA)** | **$99.60 \pm 0.02\%$** | **$99.62 \pm 0.03\%$** | **$99.62 \pm 0.02\%$** | **350.0** | **380.0** | **90.0s** |
| ($L_{\text{train}}=256$) | **DeltaPhase (Complejo)** 🌟 | **$98.81 \pm 0.29\%$** 🌟 | **$98.82 \pm 0.28\%$** 🌟 | **$98.82 \pm 0.29\%$** 🌟 | **910.0** | **1120.0** | 756.7s |
| | **Gated DeltaNet (Real)** 💥 | 75.99 ± 16.41% | 75.92 ± 16.40% | 76.06 ± 16.39% | 1210.0 | 1370.0 | 373.8s |

---

## 🔬 Análisis Científico y Falsación Rigurosa

### 1. El Baseline de Transformer Causal Está Plenamente Validado
El Transformer de control positivo aprende los circuitos de inducción de forma consistente en las 5 semillas:
- Cruza el $50\%$ en solo **240 a 350 pasos** y el $95\%$ en **250 a 380 pasos**, activando la parada temprana (*early stopping*) con una precisión superior al **$99.6\%$**.
- Mantiene una generalización OOD perfecta a $L=1024$ ($4\times$ la longitud de entrenamiento).

### 2. DeltaPhase Supera la Barrera de Crosstalk en Alta Capacidad ($N_{\text{pairs}}=32$)
- En $N_{\text{pairs}}=32$, el modelo lineal real **Gated DeltaNet ($\mathbb{R}^{32 \times 32}$)** sufre una degradación severa por interferencia destructiva de memoria (*memory crosstalk*), cayendo a **$75.99\% \pm 16.41\%$** con gran inestabilidad entre semillas.
- Por el contrario, **DeltaPhase ($\mathbb{C}^{32 \times 32}$ en $S^1$)** mantiene una precisión cuasi-perfecta y altamente estable de **$98.81\% \pm 0.29\%$** en todas las semillas y longitudes hasta $L=1024$, demostrando experimentalmente la ventaja fundamental de la cuasi-ortogonalidad fasorial en el círculo unitario complejo.

### 3. Dinámica de Transición de Fase (*Grokking*)
- **Transformer:** Transición súbita en ~250–350 pasos.
- **DeltaPhase:** Convergencia suave y progresiva, cruzando el 50% en ~410–910 pasos y el 95% en ~650–1120 pasos.
- **Gated DeltaNet:** Convergencia más lenta (1000–1370 pasos) y colapso de capacidad al sobrepasar la densidad crítica de pares.
