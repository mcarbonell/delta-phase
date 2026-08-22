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

## 🔬 Nota final sobre método

Las cuatro fronteras comparten el mismo patrón que ya funcionó dos veces en este repo: el claim original de MQAR sobrevivió a su control de capacidad solo después de recortarse a sí mismo (de "+22.82% de capacidad" a "aceleración de grokking 1.38×–1.74×"), y el kernel Triton pasó de promesa a archivado cuando el benchmark dijo que PyTorch ganaba 6/6. Las visiones de este documento merecen exactamente el mismo trato: **hipótesis baratas de escribir, caras de sostener — hasta que se miden.**
