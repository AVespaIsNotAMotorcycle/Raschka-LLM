# Raschka LLM

This repo contains the code created by working through Sebastian Raschka's book
*Build a Large-Language Model (From Scratch)*. 

## Installation

1. Clone this repo:

   `git clone git@github.com:AVespaIsNotAMotorcycle/Raschka-LLM.git`

2. Install required packages:

   `pip install -r requirements.txt`

3. Optionally, run the CLI. This lets you input prompts to which the LLM will respond:

   `python3 prompt-response.py`

## Project Structure

### GPTModel.py

This is the GPT itself, composed of `n` transformers in sequence, where `n=12` in the
model used for `prompt-response.py`. 

### Embeddings

### TransformerBlock

### LayerNorm

This takes an input `x` and transforms it such that the output, `y`, has a mean of 0
and a variance of 1. This prevents vanishing or exploding gradients during training.

### MultiHeadAttention

### FeedForward

