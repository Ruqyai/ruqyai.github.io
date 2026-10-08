---
title: "Look Inside Gemma 3 on a Free TPU"
date: 2026-10-09
permalink: /posts/2026/10/gemma3-tpu-activations/
excerpt: "Extract every layer's activations from Gemma 3 with JAX and Keras on a free TPU, then watch it answer in English before Arabic, and steer it into Arabic with one vector."
og_image: /images/posts/gemma3-tpu-activations/cover.png
tags:
  - TPU
  - JAX
  - Keras
  - Gemma 3
  - Interpretability
---

*Extracting activations with JAX and Keras, and what they reveal about Arabic and English*

![Look Inside Gemma 3 on a Free TPU](/images/posts/gemma3-tpu-activations/cover.png)

Language models feel like black boxes. You send text in, text comes out, and everything in between is "the model". But the in-between is just numbers, and you can read them. Each layer of a model hands a vector to the next one, and those vectors, the **activations**, are where the model's work happens.

In my [first TPU article](/posts/2026/06/7-reasons-to-switch-to-tpus/) I gave seven reasons to try a TPU. This one is hands-on. On a single **free TPU v5e chip in Google Colab**, we will:

1. Pull out the hidden state after **every one of Gemma 3 1B's 26 layers**, with one compiled JAX function.
2. Use the **logit lens** to watch the answer form, layer by layer.
3. Find where Gemma separates **Arabic from English**, and where it realises two sentences **mean the same thing**.
4. **Steer** the model with a single vector, and make an English prompt come back in Arabic.

Every number and chart below comes from one run of the companion notebook. **[Open the notebook](https://colab.research.google.com/github/Ruqyai/ruqyai.github.io/blob/main/_notebooks/gemma3_tpu_activations.ipynb)** and you can reproduce all of it for free.

## Why look inside at all?

Most of the time we judge a model by its answers. That tells us *what* it does, not *how*. Interpretability asks how: which layers carry which information, and what changes if we nudge them. It is also a safety tool. In an earlier article I used it to find [the single direction that controls refusal in Gemma 3](/posts/2026/04/refusal-direction-gemma-3-tpu/). The method here is the same family, applied to something friendlier: language.

## Setup: a free TPU in two minutes

In Colab, choose **Runtime → Change runtime type → v5e-1 TPU**. Accept the Gemma license on [Hugging Face](https://huggingface.co/google/gemma-3-1b-it), and add your Hugging Face token to Colab's **🔑 Secrets** as `HF_TOKEN`. (The notebook also runs on Kaggle, where it loads the same weights from Kaggle Models.)

We use **Keras 3 on the JAX backend**. JAX is the framework TPUs were designed around, and KerasHub ships Gemma 3 ready to load:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras_hub

causal_lm = keras_hub.models.Gemma3CausalLM.from_preset(
    "hf://google/gemma-3-1b-it", dtype="bfloat16")
backbone = causal_lm.backbone
tokenizer = causal_lm.preprocessor.tokenizer
```

`bfloat16` is the number format TPUs are built around. The model has **26 layers** and a hidden size of **1,152**, and it used only **7.5 GB of the runtime's 50 GB of RAM**.

## Step 1: every layer's activations

A Gemma backbone is a stack of decoder blocks. We rebuild its forward pass as a Keras functional model that **returns the output of every block**. It reuses the original layers, so the weights are shared and nothing is copied:

```python
token_ids = keras.Input(shape=(None,), dtype="int32")
padding_mask = keras.Input(shape=(None,), dtype="int32")

x = backbone.token_embedding(token_ids)
x = x * keras.ops.cast(keras.ops.sqrt(backbone.hidden_dim), x.dtype)
hidden_states = []
for block in backbone.transformer_layers:
    x = block(x, padding_mask=padding_mask)
    hidden_states.append(x)

extractor = keras.Model([token_ids, padding_mask], hidden_states)
```

Before trusting it, the notebook checks that the last block's output, after the final norm, **matches the original backbone exactly**. It does.

### Making it fast: one compiled function

`stateless_call` turns the Keras model into a pure function of *(weights, inputs)*, which is exactly what `jax.jit` wants:

```python
@jax.jit
def activations(weights, ids, mask):
    outs, _ = extractor.stateless_call(weights[0], weights[1], [ids, mask])
    h = jnp.stack(outs, axis=1).astype(jnp.float32)   # (batch, layers, seq, hidden)
    ...                                               # pool on the TPU, return small arrays
```

Two details make a big difference:

- **Pool on the TPU.** We return only each sentence's last-token state and its average over tokens, not the full tensor. Less data travels back to the host.
- **Mesh-ready.** The notebook places the weights and batch on a `jax.sharding.Mesh`. With one chip it runs as is; on an 8-core TPU (Kaggle) the same code replicates the weights and splits the batch across cores.

Here is what it cost to push **64 sentences through all 26 layers**:

![Extraction timing on TPU v5e](/images/posts/gemma3-tpu-activations/timing-en.png)

The first call takes **26.4 seconds**, almost all of it XLA compiling the program. After that the same call takes **35 milliseconds**, about **750× faster**.

The orange bar is the trap. Change the input length on every batch and JAX recompiles every time: **19.3 seconds per batch**. On TPU, **fixed shapes are everything**. Pad your inputs to a constant length and you compile once.

## Step 2: the logit lens

The last layer turns a hidden state into a prediction: final norm, then a multiplication by the embedding matrix. The **logit lens** ([nostalgebraist, 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)) applies the same read-out to **every intermediate layer**, as if the model had to answer early:

```python
@jax.jit
def lens(h):   # h: (layers, hidden)
    logits = backbone.token_embedding(backbone.layer_norm(h), reverse=True)
    return jax.nn.softmax(logits, axis=-1)
```

I asked the same question in both languages: *"The capital of Saudi Arabia is"* and *"عاصمة المملكة العربية السعودية هي"*.

![Logit lens: probability of the right answer per layer](/images/posts/gemma3-tpu-activations/logit-lens-en.png)

Three things stand out:

- **For 17 layers there is no answer at all.** The probability of the right answer stays near zero until layer 18. The early layers are busy with something else.
- **In English, "Washington" comes first.** At layer 20 the top guess is *Washington*, a generic "capital" before the specific one. One layer later it switches to **Riyadh** and never lets go, ending at **72%**.
- **In Arabic, the model passes through English.** From layer 19 to 22 the top guess for the Arabic prompt is the **English word "Riyadh"**. Only at layer 23 does it become **"الرياض"**, which it finishes with at **98%**.

That last pattern matches research on Llama-2 ([Wendler et al., 2024](https://arxiv.org/abs/2402.10588)) suggesting multilingual models work through an English-like space in their middle layers. A single prompt is an anecdote, not a proof, but it is a striking one to see on your own screen.

## Step 3: a direction for language

Next, 40 short everyday sentences, each written in Arabic and in English: *"The sun rises in the east."* / *"تشرق الشمس من الشرق."* For each sentence we average the activations over its tokens, at every layer. Then we ask two questions:

1. **Can a single direction tell the languages apart?** We take the difference between the average Arabic and the average English activation on half of the pairs, then test on the other half.
2. **Does the model know that two sentences mean the same thing?** We remove each language's average, then check whether each Arabic sentence is closest to *its own* translation among all 40. Chance is 2.5%.

![Language versus meaning, layer by layer](/images/posts/gemma3-tpu-activations/language-vs-meaning-en.png)

- **Language is always visible.** One direction separates Arabic from English with **at least 92.5% held-out accuracy at every layer**. The model never forgets which language it is reading.
- **Meaning converges, then peaks at layer 18.** Translation matching starts at **45%** in layer 1 and climbs to **100% at layer 18**. At that layer, every Arabic sentence sits closest to its own English translation.

You can see both at once in a 2-D projection:

![Sentences and their translations projected at four layers](/images/posts/gemma3-tpu-activations/pca-layers.png)

The two languages stay on opposite sides in every panel. That is the language direction. But by layer 18 the grey lines joining each sentence to its translation run far more parallel than in layer 1 or 8. The same meaning is placed at the same height on both sides, with the language offset as one shared step between them.

## Step 4: steering with one vector

If language is a direction, we can push along it. We rebuild the model once more with a third input: a vector added to the hidden state after **layer 18**. The vector is the Arabic-minus-English difference of means, multiplied by a strength `c`.

```python
for i, block in enumerate(backbone.transformer_layers):
    y = block(y, padding_mask=m_in)
    if i == STEER_LAYER:
        y = y + keras.ops.expand_dims(steer_input, 1)
```

Each row of the batch can carry its own vector, so one compiled call tries several strengths at once. The prompt is English: *"Write one short sentence about the ocean."*

| Strength `c` | Gemma 3's answer |
|---|---|
| 0 (no steering) | The ocean is a vast and powerful expanse of water covering over 70% of the Earth. |
| 0.5 | The ocean is a vast and mysterious realm teeming with life. |
| **1.0** | **المرج هو عالم البحار، مليئًا بالمعلومات والأسرار.** |
| 1.5 | broken Arabic, mixed with other scripts |
| 2.0 | broken Arabic |
| 3.0 and above | repeated fragments (…حةحةحةحة) |

At `c = 1.0` the model **answers an English question in Arabic**, and still on topic: roughly *"the world of the seas, full of knowledge and secrets."* The Arabic is not perfect: it opens with *المرج* ("meadow") and has a grammar slip. Push harder and the text falls apart. Steering has a sweet spot. Too little does nothing, and too much breaks the model. This is the same idea as activation addition ([Turner et al., 2023](https://arxiv.org/abs/2308.10248)), found here with forty sentence pairs and no training.

## Lessons for running interpretability on TPU

- **Keep shapes fixed.** A new input length means a new compilation. Here that was 19 s instead of 35 ms.
- **Reduce on the device.** Pool or slice inside the `jit` function, and send only small arrays back.
- **`stateless_call` + `jax.jit`** is the bridge from a Keras model to pure, compilable JAX.
- **Expect a flash-attention warning.** On TPU, Keras first tries the Splash attention kernel, which needs sequence lengths that are multiples of 128. With shorter inputs it logs an error and falls back to standard attention. The result is the same, and the notebook hides the messages.
- **Watch host RAM, not only TPU memory.** My first attempt crashed when the runtime ran out of RAM during loading. The final version, which skips the test `generate()` call, peaked at 7.5 GB.

## What we did

With one free TPU chip and about a hundred lines of Keras and JAX, we read all 26 layers of Gemma 3. We saw the answer appear only in the last third of the network, often in English first. We found a direction that separates the languages and a layer where their meanings align. And we used that direction to change the language of the answer.

**Try it yourself:** [open the notebook](https://colab.research.google.com/github/Ruqyai/ruqyai.github.io/blob/main/_notebooks/gemma3_tpu_activations.ipynb), swap the sentence pairs for another concept (formal vs. informal, positive vs. negative), or switch to the 4B model on a bigger TPU.

---

*Part of a short series on TPUs. Previously: [7 Reasons to Switch to TPUs](/posts/2026/06/7-reasons-to-switch-to-tpus/).*

## References

- nostalgebraist (2020). [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).
- Wendler, C., Veselovsky, V., Monea, G., & West, R. (2024). [Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://arxiv.org/abs/2402.10588).
- Turner, A. M., et al. (2023). [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248).
- Arditi, A., et al. (2024). [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717).
- [Gemma 3 1B on Hugging Face](https://huggingface.co/google/gemma-3-1b-it) · [KerasHub](https://keras.io/keras_hub/) · [JAX sharding](https://docs.jax.dev/en/latest/sharded-computation.html)
