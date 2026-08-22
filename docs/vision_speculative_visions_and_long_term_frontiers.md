# 🔮 DeltaPhase: Visiones Especulativas y Fronteras a Largo Plazo

**Tipo de documento:** Visión estratégica — **NO validado experimentalmente**
**Estado:** ESPECULATIVO — inspiración y guía de investigación
**Regla de oro del proyecto:** *medir, no confiar ni suponer; probar y falsar rápido.*

---

## ⚖️ Contrato Epistémico

Este documento **no contiene ni un solo claim validado**. Nada de lo aquí descrito debe citarse como capacidad de DeltaPhase, aparecer en benchmarks, abstracts o tablas de resultados. Su función es exclusivamente:

1. **Inspirar** la dirección de investigación a largo plazo.
2. **Guiar** la priorización de experimentos, mediante un **criterio de promoción** explícito por frontera: la condición experimental concreta que movería cada visión desde "especulativo" hacia "[POC]" o "[CORE]".

La separación es deliberada: lo validado vive en el `README.md`, los hallazgos certificados en `docs/findings_*.json/.md`, y este documento es el taller de hipótesis. Cuando un ítem pase su criterio de promoción, se migra al README con su evidencia — nunca antes.

---

## 🌌 Frontera A — Nuevos Paradigmas Computacionales sobre S¹

**Doc detallado:** [`vision_and_paradigm_breakthroughs.md`](vision_and_paradigm_breakthroughs.md)

Hipótesis central: la memoria asociativa sobre fasores unitarios no es solo *más eficiente* que la atención cuadrática, sino que habilita capacidades cualitativamente distintas (inferencia como interferencia de ondas, poda de hipótesis por cancelación destructiva, agentes de streaming continuo sin olvido catastrófico, isomorfismo fotónico).

| Capacidad visionada | Estado real hoy | Criterio de promoción a POC |
| :--- | :--- | :--- |
| Agentes de streaming 24/7 sin olvido | Estado O(1) verificado; sin benchmark de continuidad | Retención monótona en un stream continuo estilo BABILong sin replay buffer |
| Poda de hipótesis por cancelación (NOT → −1) | Operador existe en `LogicPhaseCore` [POC]; sin uso entrenado | Un modelo entrenado que demuestre invalidación medible de ramas latentes frente a control |
| Grokking zero-shot en ℤ_k | ✅ **PROMOCIONADO** — certificado en `findings_zk_grokking_rigorous_audit.md`; bloque integrado (`beta_mode='complex'`) | *(completado — ejemplo del proceso de promoción)* |
| Isomorfismo fotónico / óptico | Formalismo papel-only | Simulación de interferómetro Mach-Zehnder con fidelidad >99% vs núcleo FP32 |

---

## 🧬 Frontera B — Neural Phasor CPU (Phasor-CPU)

**Doc detallado:** [`neural_phasor_cpu_architecture.md`](neural_phasor_cpu_architecture.md)

Hipótesis central: un procesador neuro-simbólico diferenciable donde el contador de programa rota en S¹, la pila de recursión se rastrea con números de enrollamiento topológicos (w ∈ ℤ) y el heap se direcciona por resonancia holográfica.

**Estado real hoy:** formalización en papel. Cero implementación.

**Criterio de promoción a POC:** un intérprete mínimo (≤500 líneas) que ejecute programas con llamadas recursivas anidadas >100 niveles usando la pila topológica, demostrando cero corrupción de contexto bajo ruido de fase, comparado contra una pila convencional con misma tasa de perturbación.

---

## 🌊 Frontera C — Síntesis de Lenguaje por Paquetes de Onda (SpecWave)

**Doc detallado:** [`spectral_wave_language_synthesis_and_holistic_decoding.md`](spectral_wave_language_synthesis_and_holistic_decoding.md)

Hipótesis central: generar la respuesta completa como una forma de onda espectral única Ψ(ω,t) ∈ ℂ^(F×T) y decodificarla en paralelo eliminaría el cuello de botella autorregresivo token-a-token y garantizaría coherencia global argumental (subbanda LL = tesis/conclusión).

**Estado real hoy:** único fragmento implementado es el vocoder Haar 2D en `tests/test_spectral_wave_generation.py` (ida/vuelta de subbandas verificada). La generación single-shot de texto **no existe**.

**Criterio de promoción a POC:** decodificador espectral que iguale perplexidad de un baseline autorregresivo idéntico en un corpus fijo, con throughput de decodificación medido ≥5× — y análisis honesto de dónde degrada (hechos finos, sintaxis rara).

---

## 🛡️ Frontera D — Auditoría de Seguridad en Tiempo Real

**Doc detallado:** [`real_time_safety_auditing_and_mechanistic_alignment.md`](real_time_safety_auditing_and_mechanistic_alignment.md)

Hipótesis central: las propiedades físicas del estado (subbanda LL decodificable, energía hamiltoniana como tripwire, invariantes topológicos como guardarraíles) permitirían monitorizar intención interna en O(1) sin bucles auxiliares de traducción.

**Estado real hoy:** formalización en papel. Cero implementación, cero datos.

**Criterio de promoción a POC:** (1) demostrar correlación estadísticamente significativa entre la lectura de la subbanda LL e intenciones etiquetadas en un dataset construido para ello; (2) caracterizar la tasa de falsos positivos del tripwire de energía bajo distribución natural — ambos publicados con protocolo reproducible antes de cualquier claim de "detección de engaño".

---

## 🧮 Frontera E — Álgebras de Spin: Binding Cuaterniónico (S³ ⊃ U(1))

**Doc detallado:** este apartado (hipótesis formulada 22‑08‑2026 tras la auditoría completa del núcleo U(1); PoC: `tests/test_quaternion_binding_poc.py`).

Hipótesis central: el binding fasorial actual vive en el primer peldaño de la escalera clásica de grupos — U(1) (fasores, conmutativo) → SO(3)/S² (spins de Heisenberg) → SU(2)/S³ (**spinores/cuaterniones**, no conmutativo, doble recubrimiento). Subir al tercer peldaño cambia el álgebra del binding de conmutativa a **no conmutativa**, con tres consecuencias teóricas concretas:

1. **Orden estructural gratis:** en VSA, un producto conmutativo solo codifica multiconjuntos (k₁k₂ = k₂k₁); uno no conmutativo codifica **secuencias ordenadas** (trazas de camino g₁g₂g₃ ≠ g₃g₂g₁). Hipótesis operativa: desambiguación de orden sin depender de información posicional frágil bajo interferencia.
2. **Espectro de isometría enriquecido:** demostrado en este repo que β real da contracciones y β complejo isometrías exactas con autovalores en U(1) (base del grokking ℤ_k certificado). Unidades cuaterniónicas moverían los autovalores a Sp(1)=SU(2): misma estabilidad marginal, 3 grados de libertad rotacional extra por canal, composición no conmutativa.
3. **Doble recubrimiento como aritmética:** período efectivo 4π ⇒ paridad/estructuras cíclicas dobles emergentes (análogo a cómo ℤ₉ emergió de U(1)).

**Nota técnica habilitante (verificada sobre papel):** la parte real del producto interno hermitiano cuaterniónico es simétrica — Re(Σ_c k_s[c]·q̄_m[c]) define un Gram real simétrico — por lo que la maquinaria chunkwise (solve triangular, forma WY) del núcleo actual **debería sobrevivir intacta** en S³. No está verificado numéricamente.

**Estado real hoy:** formalización + PoC ejecutable (`test_quaternion_binding_poc.py`): bloque delta cuaterniónico secuencial vs brazo U(1), tareas MQAR estándar (control de sanidad) y MQAR ordenado (bigrams ordenados → valor, sensible a no-conmutatividad).

**Criterio de promoción a POC-certificado:** brazo S³ ≥ brazo U(1) fuera de ±1 SE (3 semillas) en la tarea **ordenada** bajo presupuesto de flotantes de estado igualado (~8192), con paridad en la tarea estándar como control de sanidad. Prior explícito post-ablation-del-router: se espera neutralidad salvo que la no-conmutatividad muerda algo real; cualquier victoria debe repetirse bajo control iso-presupuesto antes de migrar al README.

**Advertencias registradas de antemano:** presupuesto 4 flotantes/canal (vs 2 del complejo) — el confound de capacidad ya mordió una vez; emulación 4× ops reales sin GEMM nativo cuaterniónico (el hardware sopla en contra — lección Triton 6/6); riesgo de optimización más difícil por no-conmutatividad.

---

## 🔬 Nota final sobre método

Las cuatro fronteras comparten el mismo patrón que ya funcionó dos veces en este repo: el claim original de MQAR sobrevivió a su control de capacidad solo después de recortarse a sí mismo (de "+22.82% de capacidad" a "aceleración de grokking 1.38×–1.74×"), y el kernel Triton pasó de promesa a archivado cuando el benchmark dijo que PyTorch ganaba 6/6. Las visiones de este documento merecen exactamente el mismo trato: **hipótesis baratas de escribir, caras de sostener — hasta que se miden.**
