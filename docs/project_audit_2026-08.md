# Auditoría Integral del Proyecto DeltaPhase

**Fecha de auditoría:** 21 de agosto de 2026
**Alcance:** Idea/concepto, implementación (`delta_phase/`), suite de pruebas (`tests/`), documentación de hallazgos (`docs/`) y claims del README/paper draft.
**Método:** Lectura completa del código fuente y pruebas, verificación algebraica de las formulaciones chunkwise, y **re-ejecución independiente** de los tests clave en esta máquina.

---

## Resumen Ejecutivo

| Área | Veredicto |
| :--- | :--- |
| Formulación matemática del núcleo chunkwise | ✅ **Correcta** (verificada algebraicamente y empíricamente) |
| Implementación del bloque principal | ✅ Sólida, con reservas menores de ingeniería |
| Benchmark MQAR (comparativa principal) | 🟡 Metodología buena, pero con un **confound de capacidad sin controlar** |
| Benchmarks NIAH / gating selectivo | 🔴 Evidencia débil (aguja fija; simulación con gating oráculo) |
| Kernels Triton "fused" | 🟡 Mitigado el 2026‑08‑21 (dispatcher enruta llamadas con gradiente a la ruta PyTorch diferenciable); los kernels Triton siguen sin usarse en el benchmark de GPU y son experimentales |
| Cores secundarios (LogicPhase, Laplace) | 🟡 Conceptos interesantes, pero viven fuera de la librería, solo como scripts PoC |
| Consistencia documental | 🟡 JSON crudo consistente con README ✅, pero el paper draft contiene números contradictorios ❌ |

**Conclusión en una frase:** El núcleo técnico es real y correcto —la formulación chunkwise compleja del Delta Rule está bien derivada, bien implementada y pasa verificación de gradiente en FP64—, pero varios de los claims más llamativos del README/paper (NIAH 65K al 100%, "Triton fused", ventaja sobre Gated DeltaNet) necesitan controles experimentales adicionales o re-etiquetado honesto antes de poder sostenerse en una publicación.

---

## 1. Evaluación de la Idea

### 1.1 Linaje académico (correctamente atribuido)
El proyecto se apoya en líneas legítimas y bien citadas:
- **Delta Rule / Fast Weight Programmers** (Schlag et al. 2021) y su formulación paralela chunkwise con solve triangular (**DeltaNet**, Yang et al. 2024).
- **Gated retention / decay dependiente de datos** (Gated DeltaNet).
- **Holographic Reduced Representations / VSA con fasores FHRR** (Plate 1995): binding = producto de fasores, unbinding = conjugado.
- **SSMs / LRU**: polos complejos estables, discretización ZOH, garantía Hurwitz (LaplacePhaseCore).

La síntesis es coherente: sustituir claves/consultas reales normalizadas L2 por **fasores unitarios en S¹**, donde la afinidad `Re(K^H Q)/d_k = (1/d_k) Σ cos(θ_K − θ_Q)` juega el papel del kernel coseno.

### 1.2 Observación conceptual central (importante para interpretar resultados)
El kernel fasorial y el kernel coseno de claves reales normalizadas miden cosas muy parecidas (ambos son acotados en [−1, 1] por canal). La diferencia estructural **más grande** entre `C^{32×32}` y `R^{32×32}` no es la geometría sino que:

> El estado complejo almacena **2·d_k² flotantes reales** frente a d_k² del estado real: **el doble de capacidad bruta por cabeza**.

El propio memo técnico lo reconoce (`technical_memo_delta_phase.md`, §4), pero el README atribuye la ventaja empírica a "cuasi-ortogonalidad que mitiga crosstalk" sin descontar el efecto de capacidad. Esto se analiza en §3.1.

### 1.3 Aportes genuinos identificables
1. **Generalización compleja de la reflexión de Householder** (`β_t = 1 + e^{iφ_t}`): verificación propia — el operador `I − βkk^H` con k unitario tiene valor singular máximo `sqrt(max(1, 1 + |β|² − 2Re β))`; con `β = 1 + e^{iφ}` se cumple `Re β = |β|²/2` exactamente ⇒ **isometría exacta (σ_max = 1)**. Matemáticamente elegante: actualizaciones unitarias marginales-estables con espectro rotatorio en S¹ (vs. contracción real `|1−β|<1`). Es el aporte teórico más original del proyecto.
2. **FFN multi-sustrato espectral con router lerp aprendible** (FWHT + DCT-II + Haar): idea razonable de eficiencia paramétrica, aunque falta el ablation contra un MLP gated de presupuesto equivalente.
3. **Núcleo Laplace-Hurwitz** con σ ≤ 0 garantizado por construcción (`−softplus`) y mapeo ZOH: correcto y verificable (falsación v341 bien diseñada: forzar σ > 0 explota el estado).

---

## 2. Auditoría de la Implementación

### 2.1 Verificación matemática de `DeltaPhaseHolographicBlock.forward` (derivación propia)

La recurrencia secuencial del delta rule es `u_t = β_t(v_t − M_{t−1}k_t/d_k)`, `M_t = M_{t−1} + u_t k_t^H`. Desenrollando dentro de un chunk:

```
U = (D_β⁻¹ + tril(G, −1))⁻¹ · R ,   G[i,j] = Re⟨k_i, k_j⟩/d_k ,  R = V − K·M₀ᴴ/d_k
```

El código implementa:
```python
L_mat = triu(Gram * beta.unsqueeze(-1), diagonal=1)      # escalado de FILAS por β_i
T_mat = solve_triangular(I + L_mat.T, I, upper=False)    # (I + tril(G,−1)·diag(β))⁻¹
U_c  = beta * T_mat @ (V − KM₀ᴴ/d_k)
```
Usando la identidad `(D⁻¹ + G)⁻¹ = D(I + GD)⁻¹`, se comprueba que `diag(β)·(I + tril(G,−1)·diag(β))⁻¹ ≡ (diag(1/β)+tril(G,−1))⁻¹`. **Por tanto la vía paralela resuelve exactamente la recurrencia secuencial** — idéntica a la forma publicada de DeltaNet (Yang et al. 2024), trasladada a fasores. El escalado `inv_dk` es consistente en lectura, Gram y salida intra-chunk; el `tril` de salida incluye diagonal, coherente con `step()` (que lee después de actualizar); el manejo de padding (β=0 en posiciones rellenas, descarte de salidas) es correcto porque T es triangular unitaria.

### 2.2 Verificación empírica ejecutada durante esta auditoría

| Test ejecutado | Resultado |
| :--- | :--- |
| `tests/test_equivalence.py` (forward paralelo vs. step secuencial, L ∈ {1,63,64,65,128,1024}) | ✅ PASS — diff máx salida ≤ 3.4e−6, estado ≤ 9.1e−6 (ruido FP64→FP32 esperable) |
| Gradcheck `torch.autograd.gradcheck` FP64 sobre el bloque (reprodución independiente del claim del README) | ✅ **True** |
| `tests/test_core.py` (modelo completo: forward/backward/streaming O(1)) | ✅ PASS — 2.18M params, shapes correctas, estados encadenan |

Estos son los tres cimientos de credibilidad del proyecto y **se sostienen**.

### 2.3 Hallazgos de código

**Críticos:**
1. **`delta_phase/kernels/triton_chunk_delta.py`:**
   - `DeltaPhaseTritonFunction.backward` lanza `NotImplementedError`. Al envolver operaciones PyTorch puras en un `autograd.Function` sin backward, **cualquier intento de entrenar a través de `delta_phase_chunkwise_fused` rompe el grafo**. Riesgo alto de uso indebido por terceros.
   - El kernel `_triton_fused_phase_gram_kernel` usa bucles escalares triples (`for i… for j… for d…` con `tl.load` elemento a elemento): sería dramáticamente más lento que un matmul si llegara a usarse.
   - El kernel solo escribe entradas estrictamente inferiores de Gram; las demás quedan sin inicializar (lectura de memoria basura potencial).
2. **El benchmark de GPU no usa los kernels Triton:** inspección de `notebooks/benchmark_triton_gpu.ipynb` → 0 referencias a `DeltaPhaseTritonFunction`/`delta_phase_chunkwise_fused`, ejecución bajo `torch.no_grad()`. Los números de wall-clock (122.6K tok/s, escalado ~2× por duplicación de L) son plausibles **pero corresponden a la ruta PyTorch chunkwise pura**, no a kernels "fused Triton" como afirman README y paper. Además es benchmark solo-inferencia (sin backward).

**Menores:**
3. `LearnableSubstrateLerpFFN`: `substrate_logits` hardcodeado a 3 mientras existe el parámetro `num_banks` (=4) con otra semántica (bancos de fase, no sustratos). API confusa.
4. `forward` fuerza `theta.float()` → bloquea silenciosamente rutas AMP bf16/fp16 (costo de rendimiento no documentado).
5. `LogicPhaseCore.not_op` aloja tensores temporales en cada llamada (menor).
6. `LaplacePhaseCore` y el bloque Z_k usan bucles Python token-a-token (viola la regla propia de `GEMINI.md` de vectorización obligatoria; aceptables como PoC).
7. Versiones inconsistentes: `setup.py` = 1.0.0 vs `__init__.py` = 1.3.0; `requirements.txt` incluye `matplotlib` que `setup.py` omite.
8. **Sin `.gitignore`: los `.pyc`/`__pycache__` están commiteados** (verificado con `git ls-files`).
9. Los hacks de encoding UTF-8 de consola Windows están duplicados en cada test (candidato a utilidad compartida).

---

## 3. Auditoría de Pruebas y Benchmarks

### 3.1 MQAR riguroso (`benchmark_rigorous_mqar.py`) — la evidencia principal
**Lo bueno (bastante bueno):**
- Datos generados *on-the-fly* (inmune a memorización de dataset fijo), supervisión densa solo en posiciones de respuesta.
- 5 semillas independientes, media ± error estándar, early stopping declarado.
- Control positivo (Transformer causal MHA) y control negativo (Gated DeltaNet real) bien construidos y comparables en estilo (conv causal, pre-norm, β ∈ (0,2)).
- Extrapolación OOD de longitud (2×, 4×) y **raw JSON archivado** — verifiqué que `rigorous_mqar_results.json` coincide exactamente con las tablas del README. Integridad de datos ✅.

**El problema no resuelto — confound de capacidad:**
- En N_pairs=32 con d_k=32, el modelo real necesita almacenar 32 pares × 32 dims = 1024 valores en un estado de exactamente 1024 flotantes (capacidad al límite → crosstalk esperable), mientras el complejo dispone de 2048. El colapso del baseline real (75.99% ± 16.41% — varianza enorme, sugiere seeds divergentes) es consistente tanto con límite de capacidad **como con tuning subóptimo (lr único 3e-3 para todos los brazos)**.
- Faltan dos controles que aislarían la hipótesis "geometría fasorial ≠ solo más memoria":
  1. **Real con presupuesto de flotantes igualado** (p. ej. d_k≈45, 45² ≈ 2·32²).
  2. Ablación complejo-vs-complejo o real-vs-real con mismo byte-budget de estado.
- Nótese que en N_pairs=8/16 (lejos de saturación) la ventaja compleja existe pero es modesta (+0.79%, +1.83%) — eso sí sugiere un beneficio dinámico/geométrico real, solo que menor del que implica el titular "+22.82%".

### 3.2 NIAH — evidencia débil en sus dos variantes
1. **`test_needle_in_haystack.py`:** la aguja es **fija en todas las pruebas** (`k_needle=15, v_needle=85`). El 100% hasta 65K puede lograrse con un circuito degenerado "embedding(15) → logits(85)" que **no usa la memoria recurrente en absoluto**. No demuestra recuperación asociativa.
2. **`test_selective_gating_niah.py`:** ni siquiera es un modelo entrenado — es una simulación sintética donde la **saliencia es oráculo** (`salience[needle_pos]=1.0`, distractores 1e-4 conocidos de antemano). El "100% verde hasta 65K" mide la simulación asumiendo que ya sabes dónde está la aguja. Como demostración del mecanismo de gating es ilustrativa; como evidencia de capacidad retrieval, circular.

**Requisito mínimo para rescatar el claim:** aguja aleatoria por trial + gating β_t producido por el modelo entrenado (no oráculo) + comparación contra el mismo modelo sin gating.

### 3.3 Z_k grokking (`test_zk_group_expressivity.py`)
- La idea (β complejo ⇒ autovalores unitarios ⇒ conteo cíclico nativo) es el aporte más interesante, y la aritmética modular acumulativa es un test limpio.
- Debilidades: n=3 semillas; presupuesto fijo de 1500 pasos con lr único (el Transformer queda en 77% en Z_7 — probablemente infraentrenado, no "incapaz"); y el bloque `ComplexBetaDeltaPhaseBlock` **existe solo inline en el test**, no en la librería.
- Nota matemática positiva: verifiqué que `β = 1+e^{iφ}` produce σ_max = 1 exacto (isometría), así que la motivación teórica es sólida; falta demostrar que la ventaja persiste con presupuestos de entrenamiento igualados.

### 3.4 Otros tests
- `test_quantized_phasors_poc.py`: correcto como PoC de ALU modular uint8/uint16 con LUT; el speedup 8.12× es de microbenchmark de binding, no end-to-end.
- `test_spin_glass_recurrent_relaxation.py`, `test_pointer_augmented_memory_poc.py`, `test_spectral_wave_generation.py`: PoCs autocontenidos razonables, ninguno integrado al paquete.
- Calidad de test general: **mayoría son scripts con prints, no asserts pytest** (solo `test_equivalence` falla loudly). Sin CI. Repetición manual requerida.

### 3.5 Consistencia documental
- ✅ `docs/rigorous_mqar_results.json` ↔ tablas README: coincidencia exacta.
- ❌ `paper/paper_draft.md` afirma "MQAR 100.00% donde Transformers quedan capped at 15%" — **contradice frontalmente la auditoría rigurosa del propio repo** (99.60% Transformer vs 98.81% DeltaPhase en 32 pares). También cita "+43.58% en Z_7" donde el benchmark certificado da +33.5%. El draft está marcado SPECULATIVE (bien), pero debe reconciliarse antes de cualquier circulación.
- 🟡 Los docs de visión (fotónica, transferencia mente-a-mente, auditoría de seguridad en tiempo real) están claramente separados como speculación — aceptable, pero conviene etiquetarlos también en el índice del README.

---

## 4. Fortalezas Destacadas

1. **Matemática del núcleo correcta y verificada** — la parte difícil (WY chunkwise + solve triangular en ℂ) está bien hecha y probada en múltiples longitudes, con estados iniciales no nulos y gradientes.
2. **Cultura de trazabilidad inusualmente buena para un proyecto individual:** cabeceras con metadatos, inventarios de parámetros por brazo, niveles de rigor definidos (GEMINI.md Nivel 1/2), JSON crudo archivado, protocolos de falsación (v341: forzar σ>0 y ver explosión).
3. **Arquitectura de streaming O(1) real:** `step()` con estado conv + memoria encadena correctamente y equivale al forward paralelo.
4. Elección sensata de controles positivos/negativos en el benchmark principal.

## 5. Riesgos Principales (ordenados)

1. **R1 — Confound de capacidad** en el titular "+22.82% vs Gated DeltaNet" (§3.1). Es el riesgo de credibilidad científica mayor.
2. **R2 — Claims NIAH no demostrados end-to-end** (aguja fija / gating oráculo, §3.2).
3. **R3 — Brecha librería↔claims:** LogicPhaseCore, LaplacePhaseCore, β complejo, quantized phasors, pointer buffer existen como exports/scripts pero **ninguno está integrado en `DeltaPhaseModel` ni cubierto por tests del paquete**. Un usuario que instale `pip install delta-phase` obtiene un bloque de atención lineal compleja + FFN espectral, no las 12 "innovaciones" del README.
4. **R4 — Kernel Triton roto/misleading** (backward NotImplementedError, no usado en el benchmark, §2.3.1–2).
5. **R5 — Paper draft con números contradictorios** respecto a los propios resultados certificados (§3.5).

## 6. Recomendaciones Priorizadas — ESTADO DE REMEDIACIÓN (actualizado 21‑08‑2026)

**P0 — Antes de citar resultados públicamente:**

1. 🔴 **PENDIENTE (experimento).** Añadir a MQAR el control de capacidad igualada: Gated DeltaNet real con d_k≈45 (o 2 cabezas reales de 32) y re-tuning de lr por brazo (sweep corto 1e-3..5e-3).
2. 🔴 **PENDIENTE (experimento).** Rehacer NIAH con aguja aleatoria por trial y modelo entrenado end-to-end (gating aprendido, no oráculo).
3. ✅ **COMPLETADO (docs).** Corregir README/paper: el claim "Fused OpenAI Triton Kernels" fue renombrado a "chunkwise PyTorch" en ambos documentos con nota explicativa, y el paper draft fue reconciliado con `rigorous_mqar_results.json` (ver §8, changelog).

**P1 — Salud del código:**

4. ✅ **COMPLETADO (código + verificado).** El wrapper `autograd.Function` ya no rompe gradientes: `_chunkwise_delta_reference` fue extraída como función diferenciable y `delta_phase_chunkwise_fused` enruta automáticamente llamadas con gradiente a esa ruta; el `Function` queda reservado para inferencia con mensaje de error informativo. Verificación: backward OK, rutas numéricamente idénticas (diff = 0.0).
5. 🔴 **PENDIENTE.** Convertir tests a pytest con asserts (tolerancias explícitas) + añadir CI básico (CPU-only) que corra equivalencia + core + smoke MQAR.
6. 🔴 **PENDIENTE.** Integrar `ComplexBetaDeltaPhaseBlock` y `LaplacePhaseCore` en el paquete con tests propios, o moverlos a `experiments/` con nota clara.
7. ✅ **COMPLETADO (repo).** `.gitignore` creado (`__pycache__/`, `*.py[cod]`, artefactos de build), `.pyc` eliminados del índice de git (`git rm --cached`, archivos intactos en disco), y versión unificada a 1.3.0 en `setup.py`.

**P2 — Mejoras:**

8. 🔴 **PENDIENTE.** Soporte AMP: mantener θ en dtype del modelo o documentar la restricción FP32.
9. 🔴 **PENDIENTE.** Vectorizar `LaplacePhaseCore` (scan chunkwise análogo al bloque principal) y el kernel Gram Triton (tiles BLOCK_C × BLOCK_C con cargas vectorizadas).
10. 🔴 **PENDIENTE.** Ablation del router de sustratos: FWHT/DCT/Haar vs MLP gated de igual presupuesto, reportando probabilidades del router tras entrenamiento real.

> Nota: los ítems P0-1/P0-2 requieren ejecución de experimentos (horas de CPU/GPU), no solo edición; se mantienen como siguiente fase junto con P1-5/P1-6.

## 7. Tabla de Estado de Claims (README ↔ Evidencia)

| Claim | Estado | Comentario |
| :--- | :---: | :--- |
| Equivalencia paralelo/secuencial exacta | ✅ | Verificado aquí (≤3.4e−6 FP32; gradcheck FP64 True) |
| Gradcheck FP64 7.39e−16 | ✅ | Reproducido independientemente en esta auditoría |
| Escalado O(N), O(1) VRAM decode | ✅ | Consistente con diseño y notebook (inference-only) |
| MQAR: ventaja sobre Gated DeltaNet | 🟡 | Datos reales, pero confound 2× capacidad sin controlar |
| MQAR: "matching Softmax Transformer" | 🟡 | Cerca (98.8 vs 99.6) pero no iguala; Transformer gana en todos los bloques |
| NIAH 65K 100% | 🔴 | Aguja fija + gating oráculo; no end-to-end |
| "Fused Triton kernels" 122K tok/s | 🟡 | Wall-clock real, pero ruta PyTorch chunkwise (aclarado en README/paper); riesgo de gradiente roto mitigado el 2026‑08‑21 (dispatcher diferenciable) |
| Z_k grokking nativo | 🟡 | Mecanismo plausible (isometría verificada), baselines posiblemente infraentrenados; bloque no integrado |
| Quantized phasors 8.12× | 🟡 | Microbenchmark de binding válido; sin end-to-end |
| Laplace Hurwitz / estabilidad 100K tokens | 🟡 | Construcción correcta (σ≤0 garantizado); validación vive en scripts externos al paquete |
| Pre-entrenamiento TinyThinker-72M (PPL 26.7 @41M tokens) | ⚪ | No reproducible desde el repo (no hay código/logs de ese run aquí) |

---

## 8. Changelog de Remediación (21‑08‑2026)

Remediación aplicada tras esta auditoría, en dos fases: documentación primero, luego código.

### Fase 1 — Documentación

**`paper/paper_draft.md`** (reconciliado con los resultados certificados del propio repo):
- **Abstract:** reemplazados los claims "100.00% MQAR / Transformers capped at 15%" y "+43.58% en Z_7" por los números certificados Nivel 2 (98.81% ± 0.29% vs Transformer 99.60%, +22.82% vs Gated DeltaNet; +33.50% en Z_7), con caveat explícito del confound de capacidad (2× flotantes).
- **§3.2:** aclarado que el núcleo publicado usa β real ∈ (0,2) y que la variante Householder compleja β = 1+e^{iφ} vive en `tests/test_zk_group_expressivity.py` (pendiente de integración).
- **§3.4:** corregida la discretización — el código implementa **ZOH** (`z = e^{sΔt}`), no transformada bilineal.
- **§3.3:** notación precisa del escalado `diag(β_c)·G` (escalado por filas), consistente con la derivación verificada.
- **§5.1:** eliminada la tabla v349 (baseline Transformer roto: 15%) y sustituida por la tabla certificada multi-semilla + caveat de capacidad.
- **§5.2:** números antiguos single-run (67.89%/23.70%) sustituidos por certificados (Z_7/Z_9/Z_12, 3 seeds) + caveat de presupuesto de entrenamiento fijo.
- **§5.3:** re-titulado "Selective-Gating NIAH **Simulation**" con nota de alcance: saliencia oráculo, no retrieval end-to-end.
- **§5.4:** "fused OpenAI Triton kernel" → "vectorized parallel chunkwise PyTorch implementation" (forward-only).
- **Apéndice A.1:** prueba del Teorema 1 reforzada — los autovalores no bastan para acotar la norma de operadores no normales; se añadió el argumento de valores singulares: σ_max(H)² = max(1, 1+|β|²−2Re β), que con β = 1+e^{iφ} da σ_max = 1 exacto (isometría marginal) y con β real ∈ (0,2) contracción estricta.
- **Apéndice B:** comandos de reproducción corregidos (`scratch/run_head_to_head_dk32.py` no existe → `tests/test_zk_group_expressivity.py`; MQAR → `benchmark_rigorous_mqar.py`; añadidos tests de equivalencia).

**`README.md`:**
- Claim de contribución actualizado (+0.79% / +1.83% / +22.82% según datos certificados, antes "+3.4% a +5.9%") con ⚠️ caveat de capacidad añadido.
- Leyenda de estado [CORE] / [POC] / [VISIÓN] añadida y aplicada a las 12 secciones de innovaciones (deja claro qué está integrado en la librería, qué es PoC en scripts y qué es especulativo).
- §2 corregida: el núcleo implementado NO usa atenuación λ_t (esa variante gated vive en `LaplacePhaseCore`) — la descripción anterior atribuía al bloque principal un mecanismo que no tiene.
- §Benchmarks-4: columna "DeltaPhase Fused" → "DeltaPhase Chunkwise" + nota de precisión sobre la ruta medida (PyTorch forward-only, sin Triton).
- §Benchmarks-5: NIAH re-etiquetado como simulación con saliencia oráculo + alcance pendiente end-to-end.
- Tabla "Architectural Completeness": estados ajustados a la evidencia real (filas 4 y 5 pasan de ✅ Verified a 🟡 micro-benchmark / simulación; fila 6 actualizada a +33.5%).

### Fase 2 — Repositorio y código

- **`.gitignore` creado** (`__pycache__/`, `*.py[cod]`, `build/`, `dist/`, entornos, artefactos de experimentos).
- **`.pyc` eliminados del índice de git** vía `git rm -r --cached` (los archivos permanecen en disco; solo deja de trackearlos).
- **`setup.py`:** versión 1.0.0 → **1.3.0**, unificada con `delta_phase/__init__.py`.
- **`delta_phase/kernels/triton_chunk_delta.py` — fix de gradiente (R4):**
  - Extraída `_chunkwise_delta_reference(...)`: la ruta chunkwise PyTorch como función pura **totalmente diferenciable** por autograd nativo.
  - `DeltaPhaseTritonFunction.forward` ahora delega en ella; su `backward` lanza `NotImplementedError` con mensaje accionable (usar el dispatcher para entrenar).
  - Nuevo dispatcher `delta_phase_chunkwise_fused`: con gradientes habilitados usa la ruta diferenciable automáticamente; bajo `torch.no_grad()` usa el wrapper (futuros kernels fused). Ambas rutas numéricamente idénticas.
  - **Verificación ejecutada:** backward OK (grad fluye), ruta no-grad OK, diff grad-vs-nograd = 0.0.

### Pendiente (siguiente fase)
- P0-1/P0-2: experimentos de control (capacidad igualada en MQAR; NIAH end-to-end con aguja aleatoria).
- P1-5: migración de tests a pytest + CI.
- P1-6: integración (o reubicación en `experiments/`) de `ComplexBetaDeltaPhaseBlock` / `LaplacePhaseCore`.
- P2: AMP, vectorización de `LaplacePhaseCore`, kernel Gram Triton con tiles, ablation del router FFN.

---

*Auditoría realizada mediante revisión estática del 100% del código fuente y pruebas, verificación algebraica independiente de la formulación chunkwise, inspección del notebook de GPU y del JSON crudo de resultados, y ejecución local de `test_equivalence.py`, `test_core.py` y gradcheck FP64. Remediación documental/código aplicada y verificada el 21‑08‑2026.*
