## 🧠 Character-Level Language Model using GPT Architecture

### 📄 Overview
This project implements a GPT-style Transformer model to generate character-level text using self-attention, trained on a custom dataset. The model learns to predict the next character in a sequence and can generate long coherent text from a given context. It is a well structed implementation enhanced with modular architecture, training evaluation, and model serialization.

### 📁 Project Structure

Bigram_Language_Model/
│
├── input.txt                  # Raw training text data
├── GPT_dataPrep.py           # Data preprocessing, encoding/decoding, batching
├── GPT_block.py              # Transformer-based GPT model definition
├── train_eval.py             # Training and evaluation loop
├── generate.py               # Text generation script using the trained model
├── hyper_parameters.py       # Centralized configuration for hyperparameters
├── GPT.pth                   # Saved PyTorch model checkpoint
├── GPT_output_test.txt       # Generated output text from the model

### ⚙️ Hyperparameters (hyper_parameters.py)

NUM_EPOCHS = 5000
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
BLOCK_SIZE = 256
NUM_EMBED = 384
NUM_HEADS = 6
NUM_LAYERS = 6
DROPOUT = 0.2
MAX_NEW_TOKENS = 10000
TRAIN_SIZE = 0.90

### 🔄 Data Preparation (GPT_dataPrep.py)

Loads and encodes the character dataset from input.txt

Creates a character-level vocabulary

Implements encode, decode, and get_batch for sequence sampling

Example Output Shapes:
Input Tensor: (BATCH_SIZE, BLOCK_SIZE)

Target Tensor: (BATCH_SIZE, BLOCK_SIZE)

### 🧱 Model Architecture (GPT_block.py)

Key Components:
Embedding Layer — token + positional embeddings

Multi-Head Self-Attention — implemented using custom Head, MultiHeadedAttention

### 🏋️‍♂️ Training Loop (train_eval.py)
Trains on batches sampled from the dataset.

Evaluates every epoch on validation data.

Saves the best model based on lowest validation loss.

Logs losses every 500 epochs.


FeedForward Layers — Two-layer MLP with ReLU

Residual + LayerNorm — for stability

Output Projection Layer — maps to vocab size

### 🔮 Text Generation (generate.py)
Loads trained model weights from GPT.pth

Accepts user input as seed text or uses newline token

Outputs generated text to GPT_output_test.txt
