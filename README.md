# gpt-from-scratch

A small character-level GPT built from scratch in PyTorch: self-attention, Transformer blocks, and a training loop, trained on Tiny Shakespeare.

## What it does

This project builds a GPT-style language model from the ground up to understand how models like ChatGPT work inside. The model reads text one character at a time and learns to **predict the next character**. After training on ~1.1 million characters of Shakespeare, it can write new text in a Shakespeare-like style: character names, line breaks, and mostly real words.

It's a learning project. The goal isn't useful text, but a working understanding of the Transformer architecture behind modern large language models.

## How it works

| Step | What it does |
|---|---|
| **1. Data** | Downloads Tiny Shakespeare (~1.1M characters) |
| **2. Tokenizer** | Maps each of the 65 unique characters to a number (`encode` / `decode`) |
| **3. Training examples** | Splits the data 90% train / 10% test; each target is the *next* character |
| **4. Self-attention** | Each position builds a **Query**, **Key** and **Value**; a causal mask stops it from seeing the future |
| **5. Transformer blocks** | Multi-head attention + feed-forward layer, with residual connections and layer norm |
| **6. Training loop** | AdamW optimizer, cross-entropy loss, 3,000 steps |
| **7. Generation** | Samples one character at a time to write new text |

**Model size:** ~0.82M parameters. 4 layers, 4 attention heads, 128-dim embeddings, 64-character context window.

## Results

The training and test loss both fall steadily, from about **4.3** at the start (random guessing across 65 characters) to under **1.9** within the first ~1,000 steps.

## Run it yourself

1. Open `gpt-from-scratch.ipynb` in [Google Colab](https://colab.research.google.com/).
2. (Optional, much faster) **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**.

Training takes about 1–2 minutes on a GPU, or about 20 minutes on a CPU.

## Tech

Python · PyTorch · Google Colab

## Credits

Based on Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial and the Tiny Shakespeare dataset from [karpathy/char-rnn](https://github.com/karpathy/char-rnn).
