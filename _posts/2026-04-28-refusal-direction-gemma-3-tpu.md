---
title: "Exploring the Refusal Direction in Gemma 3 Models on TPU"
date: 2026-04-28
permalink: /posts/2026/04/refusal-direction-gemma-3-tpu/
excerpt: 'Applying the method from "Refusal in Language Models Is Mediated by a Single Direction" to Google''s Gemma 3 family, with experiments accelerated on TPUs.'
tags:
  - Interpretability
  - AI Safety
  - Gemma 3
  - TPU
---

*Exploring the Refusal Direction in Gemma 3 Models on TPU*

**An extension of “Refusal in Language Models Is Mediated by a Single Direction”**

*Applying the original methodology to Google’s new Gemma 3 family while accelerating experiments with Tensor Processing Units*

**Original paper:** [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) **Code:** [GitHub — andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) **Original blog post:** [LessWrong](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)  
**My fork:** [Ruqyai/refusal_direction](https://github.com/Ruqyai/refusal_direction)

## Why This Research Matters

In April 2024, Andy Arditi and his team from the MATS program published a groundbreaking study that revealed a striking finding: **refusal behavior in large language models is controlled by a single direction in the internal representation space**. This means that all the varied responses we see when a model refuses a harmful request — from “I’m sorry, I can’t help with that” to “This request is inappropriate” — all pass through a single bottleneck in the neural network.

> ***Original Paper*** *“Refusal in Language Models Is Mediated by a Single Direction” — Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda. Published as part of the ML Alignment & Theory Scholars (MATS) program under the supervision of Neel Nanda and Wes Gurnee.*

## Key Findings of the Original Research

The researchers demonstrated that through a simple computation — taking the difference between mean activations on harmful instructions and harmless instructions — a “refusal direction” can be extracted from any language model. When this direction is removed (an operation called ablation), the model completely loses its ability to refuse. The reverse is also true: when this direction is injected into harmless instructions, the model begins refusing them as if they were harmful.

This phenomenon was proven across multiple model families including Llama 2, Llama 3, Gemma, Qwen, and Yi, and across different model scales. This indicates that the refusal mechanism is a general property rather than a model-specific trait.

The ablation formula is straightforward:

```text
c'_out ← c_out − (c_out · r̂) r̂
```

Where r̂ is the normalized refusal direction and c_out is the output of any component writing to the residual stream.

## Scientific and Practical Value

This research carries dual value. From a scientific perspective, it provides strong evidence that language models use linear representations to encode complex concepts like “should refuse.” This supports the linear representation hypothesis in the field of mechanistic interpretability and opens the door to a deeper understanding of how these models work internally.

From a practical standpoint, the research reveals the fragility of safety mechanisms in open-source models. Instead of requiring expensive fine-tuning or complex attacks, the safety guardrails can be bypassed with a simple weight modification. This doesn’t pose a new risk per se — it was already known that safety guardrails could be removed through fine-tuning — but it underscores the need for deeper and more robust safety approaches.

## Why Explore the Gemma 3 Family?

The original research studied Gemma 2B-IT (first generation) and confirmed the existence of the refusal direction. However, the Gemma 3 family represents a significant architectural leap that warrants independent study. The new version (March 2025) introduces several fundamental improvements that make us ask: will the same phenomenon still apply?

### What’s New in Gemma 3?

Gemma 3–1B-IT differs from its predecessor in several key ways. First, it uses a hybrid architecture of self-attention layers that alternates between local sliding windows and global attention, at a ratio of 5 local layers for every global layer. Second, it supports a context length of 128K tokens, a massive increase compared to the previous generation. Third, it was trained using knowledge distillation and reinforcement learning techniques, which may affect how refusal behavior is encoded.

> ***Research Hypothesis*** *We expect the refusal direction to still exist in Gemma 3–1B-IT given the universality of the phenomenon across different architectures. However, the hybrid attention structure may affect which layer the direction appears most strongly in. The direction may also be sharper or more diffuse as a result of the new training techniques.*

## The Experimental Pipeline

The pipeline follows the same five steps from the original research, with modifications to support TPU and the Gemma 3 architecture.

**Step 1 — Extract Candidate Refusal Directions** Compute the difference in means between harmful and harmless activations at each layer.

**Step 2 — Select the Best Refusal Direction** Evaluate each candidate direction on a validation set to select the most effective one.

**Step 3 — Evaluate Bypass Refusal** Remove the direction and measure the decrease in refusal rate on harmful instructions.

**Step 4 — Evaluate Induced Refusal** Inject the direction into harmless instructions and measure the increase in refusal rate.

**Step 5 — Evaluate CE Loss Impact** Measure how much the ablation affects the model’s general language capabilities.

## Setup and Execution

```bash
# 1. Clone the original repository
git clone https://github.com/Ruqyai/refusal_direction.git
cd refusal_direction

# 2. Install requirements + TPU support
pip install -r requirements.txt
pip install torch_xla cloud-tpu-client

# 3. Run the modified pipeline for Gemma 3
export HF_TOKEN="your_huggingface_token"
python refusal_direction_gemma3_tpu.py \
    --model_path google/gemma-3-1b-it \
    --n_instructions 256 \
    --batch_size 8
```

## Key Modifications to the Original Code

The original code required several modifications to work with Gemma 3 on TPU. First, torch.cuda calls were replaced with the appropriate torch_xla equivalents, with synchronization points (xm.mark_step()) added at the right places to ensure efficient execution on TPU. Second, the chat template was updated to use Gemma 3's specific format (\<start_of_turn\>). Third, bfloat16 was chosen as the default data type because it is natively supported on TPU and provides better performance than float16.

```python
# Automatic TPU detection with GPU/CPU fallback
import torch_xla.core.xla_model as xm
DEVICE = xm.xla_device()

# Load model in bfloat16 (native on TPU)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    torch_dtype=torch.bfloat16,
    device_map=None,
).to(DEVICE)

# TPU synchronization after batches
if TPU_AVAILABLE:
    xm.mark_step()
```

## Accelerating Interpretability Experiments with TPU

![](/images/posts/refusal-gemma3/1_8HctV04k9nVaGLvlFlLC7g.png)

I picked **“Refusal in Language Models Is Mediated by a Single Direction”** (Arditi et al., 2024) as the experimental ground, and applied it to **Gemma 3–1B-IT** — Google’s latest open-weight model family. Accelerating Interpretability Experiments with TPU

Mechanistic interpretability experiments require a significant amount of computation. Extracting activations from hundreds of instructions across all model layers, then evaluating dozens of candidate directions by generating full text completions — all of this demands substantial computational resources. This is where the real value of TPU shines.

### Why TPU Is Well-Suited for This Type of Experiment

TPUs have several characteristics that make them ideal for interpretability experiments. First, they use systolic arrays designed specifically for the matrix multiplication operations that form the backbone of the forward pass in neural networks. Second, they feature high-bandwidth memory (HBM) that allows loading the entire model and storing intermediate activations efficiently. Third, the native support for bfloat16 is a significant advantage because it preserves dynamic range while halving memory footprint.

> ***Practical Tip*** *When working on TPU, use* *xm.mark_step() judiciously. Calling it too frequently prevents efficient XLA graph compilation, while calling it too rarely may cause memory buildup. A practical rule of thumb: synchronize every 5–10 batches.*

### Performance Optimizations

The accompanying code includes several TPU-specific optimizations. The inference loop is compiled as a single XLA graph, reducing host-device communication overhead. The batch size is set to multiples of 8 to maximize bus bandwidth utilization on TPU v3 units. Additionally, extracted activations are moved directly to CPU (.cpu()) to avoid exhausting TPU memory during the extraction phase.

```python
# Extracting activations with TPU optimizations
for batch_idx in range(n_batches):
    with torch.no_grad():
        _ = model(**inputs)

    # Periodic sync — not every batch
    if TPU_AVAILABLE and (batch_idx + 1) % 10 == 0:
        xm.mark_step()

# Move activations to CPU immediately to save TPU memory
    for layer_idx in range(n_layers):
        last_acts = activation_cache[layer_idx]
        layer_activations[layer_idx].append(last_acts.cpu())
```

![](/images/posts/refusal-gemma3/1_RtzFOd11aPKRvOSPS7Smxg.png)

![](/images/posts/refusal-gemma3/1_585r-kz0fcfJ7PG4u9x60g.png)

![](/images/posts/refusal-gemma3/1_2CyTzzh5WZTjfhk9jiKiBQ.png)

## TPU vs GPU for Interpretability Experiments

Researchers in mechanistic interpretability face an important choice when selecting hardware. Both TPU and GPU have advantages and drawbacks that must be understood in the context of this specific type of experiment. Below is a comprehensive comparison based on practical experience and available benchmarks.

| Criterion | TPU (v3/v4/v5) | GPU (A100/H100) |
|---|---|---|
| **Compute architecture** | Systolic arrays specialized for matrix operations | Versatile CUDA cores with Tensor Cores |
| **Raw compute performance** | TPU v4: up to 275 TFLOPS, highest throughput for matrix ops | A100: up to 312 TFLOPS (with sparsity); H100: up to 989 TFLOPS |
| **bfloat16 support** | Native and optimized, ideal for inference and training | Supported on Ampere+, but float16 is more common in practice |
| **Memory (HBM)** | TPU v4: 32 GB, v5e: 16 GB, sufficient for 1B–7B models | A100: 40/80 GB, H100: 80 GB, larger and more flexible |
| **Hourly cost (cloud)** | ~1.35–3.22 USD/hr (v3-8), lower cost per FLOP | ~3.67–12.00 USD/hr (A100/H100), higher but more widely available |
| **Software ecosystem** | JAX native, PyTorch via XLA, steeper learning curve | Mature CUDA, native PyTorch, broad support and abundant tooling |
| **Availability** | Google Cloud only, plus Kaggle/Colab (limited free tier) | Available on all cloud providers and for on-premise purchase |
| **Fit for interpretability** | Excellent for bulk activation extraction and repeated forward passes | Excellent for interactive experiments, larger models, and analytical tools |
| **Debugging ease** | Harder: lazy execution makes tracing less straightforward | Easier: eager mode facilitates debugging |

### When to Choose Each

The optimal choice depends on the nature of the experiment. If you are running interactive exploratory experiments — such as inspecting individual activations or quickly testing different hypotheses — GPU with PyTorch in eager mode is the better choice. If you are running a full pipeline that requires processing hundreds or thousands of instructions across all model layers (as in this research), TPU offers a better performance-to-cost ratio thanks to its optimization for repetitive matrix operations.

A hybrid approach is optimal in practice: use GPU for development, debugging, and quick experiments, then switch to TPU to run the full pipeline at scale. The accompanying code is designed to support this pattern — it automatically detects TPU availability and falls back gracefully to GPU or CPU.

## Expected Results and Analysis

Based on the original research results on Gemma 2B-IT and other models, we expect our experiments on Gemma 3–1B-IT to show the following pattern: a sharp drop in refusal rate when the refusal direction is removed (from ~90% to below 10%), and a notable increase in refusal rate on harmless instructions when the direction is injected (from ~0% to over 70%). We also expect the impact on CE loss to be limited, indicating that removing the refusal direction does not significantly harm general language capabilities.

## Open Research Questions

This extension raises several questions worth exploring. Does the alternation between local and global attention in Gemma 3 affect where the refusal direction appears? In previous models, the direction appeared most strongly in the middle-to-late layers. But with the hybrid architecture, we may find that global attention layers (which appear every 6 layers) play a larger role in encoding the refusal decision.

Another important question: did the knowledge distillation from a larger model (used in training Gemma 3) affect the “clarity” of the refusal direction? Distillation may have produced a more distributed representation of refusal behavior, making the single direction less effective. Or the opposite could be true — perhaps distillation concentrated refusal behavior into a single, more distinct direction.

> ***Ethical Note*** *This research aims to understand and improve safety mechanisms in language models. Removing the refusal direction produces an unsafe model that should not be deployed or used in production. The purpose is purely research-oriented, aimed at developing stronger and deeper safety mechanisms.*

## Conclusion and Next Steps

Exploring the refusal direction in Gemma 3 on TPU is a natural extension of the pioneering original research. Combining a new architecture (Gemma 3) with specialized hardware (TPU) allows us to test the generality of the original findings while benefiting from higher computational efficiency. The accompanying code is ready to run on any available TPU environment, including Google Colab (free TPU) and Google Cloud TPU VMs.

For researchers interested in going further, we recommend trying larger sizes from the Gemma 3 family (4B, 12B, and 27B) to study how the refusal direction changes with model scale. It would also be valuable to compare the refusal direction in Gemma 3 with other recent models like Llama 3.2 and Qwen 2.5 to explore whether modern training techniques have changed the nature of this direction.

**This is an updated fork of the repository:**  
<https://github.com/Ruqyai/refusal_direction/blob/main/Gemma%203-1B-IT%20on%20TPU.md>

**Resources:** [Original Paper (arXiv)](https://arxiv.org/abs/2406.11717) · [Original Code (GitHub)](https://github.com/andyrdt/refusal_direction) · [LessWrong Post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) · [Gemma 3–1B-IT (HuggingFace)](https://huggingface.co/google/gemma-3-1b-it)

*Applied research building on* [*Arditi et al., 2024*](https://arxiv.org/abs/2406.11717)*. Code licensed under Apache 2.0.* *Gemma 3 is a model by Google. TPU is a trademark of Google.*

## Thanks to Google Cloud

**Google Cloud credits are provided for this project.** #TPUSprint

---

*Also published on [Medium](https://medium.com/@rbinsafi/exploring-the-refusal-direction-in-gemma-3-models-on-tpu-3c9858f7426b).*
