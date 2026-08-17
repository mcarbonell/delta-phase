# Informe de Hallazgos: Experimento v349 - Exploratorio Inicial DeltaPhase en MQAR

> [!NOTE]
> **ESTADO DEL DOCUMENTO: [EXPLORATORIO / SUPERSEDIDO]**
> Este informe corresponde al sondeo preliminar de Nivel 1. Ha sido formalmente auditado y superseded por el protocolo estandarizado multi-semilla on-the-fly de Nivel 2 en [`docs/findings_mqar_rigorous_audit.md`](findings_mqar_rigorous_audit.md), donde tanto DeltaPhase como el Transformer Causal alcanzan >99% de precisión con controles positivos y negativos rigurosos.

**ID Experimento:** v349_deltaphase  
**Fecha:** 13 de Agosto, 2026  
**Proyecto:** DeltaPhase  
**Ubicación:** `C:\Users\mrcm_\Local\proj\algorithms\delta-phase\docs\findings_v349_deltaphase_100pct_mqar.md`

---

## 0. Resumen Ejecutivo

En este experimento se evalúa **DeltaPhase** (bloque matricial de fase compleja $\mathbb{C}^{d_k \times d_k}$ en $S^1$ con solucionador chunkwise WY `solve_triangular`, Convolución Causal 1D $k=4$ y FFN Espectral Multisustrato) en la tarea sintética asociativa estandarizada **MQAR (Multi-Query Associative Recall)**.

* **Resultado Absoluto:** **100.00% de Precisión en todas las longitudes ($L=128, 256, 512$)** con una pérdida residual de solo **$0.0007$ nats**.
* **Superación del Transformer:** El Transformer de Anthropic (`Causal Induction Transformer`) se estancó en el **15.00%-17.50%** de precisión bajo las mismas condiciones.

---

## 1. Resultados Empíricos del Experimento v349

### 1.1. Métricas de Entrenamiento y Eficiencia

| Modelo / Arquitectura | Parámetros Totales | Loss Final (Época 30) 🌟 | Wall Clock Time (s) |
| :--- | :---: | :---: | :---: |
| **DeltaPhase Holographic Core ($\mathbb{C}^{32 \times 32}$)** 🌟 | 296,014 | **0.0007** 🌟 | 2830.82s |
| **Causal Induction Transformer (Anthropic Circuit)** | **281,408** 🌟 | 2.9391 | **1269.22s** 🌟 |

### 1.2. Precisión MQAR Zero-Shot por Longitud de Secuencia ($L$)

| Modelo / Arquitectura | $L=128$ (Train) 🌟 | $L=256$ Zero-Shot 🌟 | $L=512$ Zero-Shot 🌟 |
| :--- | :---: | :---: | :---: |
| **DeltaPhase Holographic Core ($\mathbb{C}^{32 \times 32}$)** 🌟 | **100.00%** 🌟 | **100.00%** 🌟 | **100.00%** 🌟 |
| **Causal Induction Transformer (Anthropic Circuit)** | 15.25% | 17.50% | 15.00% |

---

## 2. Pilares de la Resolución de Memoria Asociativa

1. **Memoria de Estado Matricial de Fase Compleja ($\mathbb{C}^{32 \times 32}$):**  
   Proporciona 2,048 flotantes reales de memoria por cabeza con alta quasi-ortogonalidad en $S^1$, eliminando la interferencia entre claves.
2. **Regla Delta de Corrección de Errores:**  
   $E_c = T_{\text{mat}} (V_c - V_{\text{old}})$. Solo escribe la parte no aprendida del error, previniendo la sobreescritura del estado.
3. **Invariancia de Longitud Extrema:**  
   Generaliza perfectamente a $L=512$ zero-shot sin sufrir atenuación de señal o distorsión logarítmica.
