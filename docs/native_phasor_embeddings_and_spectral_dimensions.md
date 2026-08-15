# 🌌 Embeddings Nativos de Fasores en $(S^1)^D$ y Compresión Espectral de la Dimensión $D$

**Documento:** Propuesta de Diseño Arquitectónico y Especificación Matemática  
**Fecha:** 15 de Agosto, 2026  
**Estado:** Propuesta Activa & Hoja de Ruta para Futura Implementación  

---

## 1. El Planteamiento: Superar los Embeddings Euclídeos Reales

En las arquitecturas tradicionales de Deep Learning, los diccionarios de embeddings asignan a cada palabra un vector plano en el espacio euclídeo:
$$\vec{x}_w \in \mathbb{R}^D \quad \text{con magnitudes arbitrarias } \|\vec{x}_w\|_2 \in (0, \infty)$$

Este enfoque introduce **tres problemas fundamentales**:
1. **Inestabilidad de Magnitud:** Algunas palabras tienen normas gigantes y otras minúsculas, distorsionando la atención y requiriendo capas continuas de *LayerNorm / RMSNorm*.
2. **Deformación por Suma Vectorial:** Combinar embeddings mediante suma lineal ($\vec{x}_{\text{adj}} + \vec{x}_{\text{sust}}$) altera la longitud del vector e introduce interferencia no ortogonal irreversible.
3. **Consumo Masivo de VRAM:** Una tabla de $32.000$ palabras a dimensión $D=1024$ en `FP16` ocupa **$65.5\text{ MB}$** de memoria pesada en GPU.

---

## 2. Solución 1: Embeddings Nativos de Fasores en $(S^1)^D$ (La Geometría Circular)

En lugar de vectores euclídeos, **cada palabra del vocabulario se define como un vector de $D$ ángulos de fase unitarios en el círculo complejo**:

$$\vec{\theta}_w = \left[ \theta_1, \theta_2, \dots, \theta_D \right] \in [0, 2\pi)^D \quad \Longrightarrow \quad \vec{E}_w = e^{i \vec{\theta}_w} \in (S^1)^D$$

```
     EMBEDDING EUCLÍDEO TRADICIONAL (ℝᴰ)              EMBEDDING DE FASORES DELTAPHASE (S¹)ᴰ
   ┌─────────────────────────────────────┐          ┌──────────────────────────────────────┐
   │ • Magnitud variable: ||v|| = 0.2..5 │   ──►    │ • Norma unitaria estricta: |e^iθ| = 1│
   │ • Suma deforma el espacio semántico │          │ • Composición por suma modular (2π)  │
   │ • Requiere FP16/FP32 (2-4 bytes/dim)│          │ • Almacenable en enteros uint8 (1B)  │
   └─────────────────────────────────────┘          └──────────────────────────────────────┘
```

### 🌟 Propiedades y Ventajas Matemáticas:

1. **Isometría Universal (Norma = 1.0 para Siempre):**
   * Cada canal del embedding satisface estrictamente $|e^{i\theta_j}| = 1$.
   * Se elimina la necesidad de normalizaciones artificiales y se previene la explosión de gradientes desde la primera capa.
2. **Álgebra Holográfica FHRR Exacta (Binding y Unbinding Reversibles):**
   * **Vincular dos conceptos (Binding):** Es una suma modular de ángulos en el toro $T^D$:
     $$\vec{\theta}_{\text{concepto}} = (\vec{\theta}_{\text{adjetivo}} + \vec{\theta}_{\text{sustantivo}}) \pmod{2\pi}$$
   * **Desvincular y Recuperar (Unbinding):** Es una resta modular exacta sin pérdida de señal:
     $$\vec{\theta}_{\text{sustantivo}} = (\vec{\theta}_{\text{concepto}} - \vec{\theta}_{\text{adjetivo}}) \pmod{2\pi}$$
3. **Cuantización Nativa a `uint8` ($75\%$ Ahorro de VRAM):**
   * Al mapear $[0, 2\pi) \to [0, 255]$, la tabla de embeddings ocupa **1 solo byte por dimensión**.
   * Una tabla de $32.000$ tokens con $D=1024$ pesa únicamente **$32.7\text{ MB}$**.
   * La vinculación se ejecuta en la **ALU de enteros de la CPU/GPU con suma modular por hardware**: `(a + b) & 0xFF` (0 ciclos adicionales).
4. **Fusión Directa con el Núcleo DeltaPhase:**
   * Al ser ya fasores unitarios, no se requieren matrices lineales $W_K, W_Q$ para convertir vectores reales en ángulos: la entrada ya viaja en el formato nativo de las ondas armónicas.

---

## 3. Solución 2: Compresión Espectral de la Dimensión $D$ (Vía 1D-DCT)

Inspirado en el **Experimento v370** (donde 64 coeficientes DCT superaron a 784 píxeles), la tabla de embeddings puede comprimirse aprendiendo únicamente sus **$K$ componentes de baja frecuencia ($K \ll D$)**:

$$\vec{E}_{D} = \text{IDCT}_{1D}(\vec{C}_K), \quad K \in [32, 64]$$

* **Fundamento:** Investigaciones lingüísticas demuestran que la dimensión intrínseca del lenguaje ronda las $30\text{ a }60$ dimensiones semánticas.
* **Ventaja:**
  * Reduce la tabla de embeddings en un factor de **$16\times$ a $32\times$**.
  * Actúa como regularizador espectral de baja frecuencia, impidiendo que palabras raras memoricen ruido ortogonal espurio.

---

## 4. Matriz Comparativa de Paradigmas

| Característica | Embeddings Tradicionales (PyTorch `nn.Embedding`) | Embeddings de Fasores $(S^1)^D$ (`delta_phase`) |
| :--- | :--- | :--- |
| **Dominio** | Espacio Euclídeo Plano $\mathbb{R}^D$ | Toro Complejo Compacto $(S^1)^D \cong \mathbb{T}^D$ |
| **Tipo de Dato** | `FP16` / `FP32` (2 a 4 bytes/dim) | **`uint8` (1 byte/dim)** |
| **Operación de Vinculación** | Suma lineal / Concatenación | **Suma modular de fase $(a + b) \pmod{256}$** |
| **Inversión / Recuperación** | Inexacta / Aproximada | **Exacta mediante resta modular $(a - b) \pmod{256}$** |
| **Norma del Vector** | Variable e Inestable ($\|\vec{x}\| \neq 1$) | **Estrictamente Isométrica ($|e^{i\theta}| = 1$)** |
| **Compatibilidad Hardware** | Tensor Cores FP16 | **ALU Entera 8-bit, Fotónica & Óptica Coherente** |

---

## 5. Próximos Pasos Experimentales

1. **Diseñar el módulo `PhasorEmbedding(vocab_size, d_model)`** con lookup `uint8` y operaciones de binding/unbinding en silicio.
2. **Evaluar en benchmarks de analogías léxicas (Word2Vec Analogies):** Medir la precisión de `Rey - Hombre + Mujer = Reina` en aritmética modular de fasores frente a vectores reales.
3. **Integración End-to-End con el modelo DeltaPhase:** Eliminar las proyecciones intermedias de entrada para alimentar fases puras al núcleo recurrente.
