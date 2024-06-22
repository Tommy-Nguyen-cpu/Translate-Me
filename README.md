# Translate-Me
Developed a translation model utilizing the transformer architecture.

## Transformer Architecture
The Transformer architecture was built from scratch without using the built EncoderLayer and DecoderLayer provided by TensorFlow.
The Transformer architecture is currently simplified and does not entirely follow the Transformer architecture provided in the paper "Attention Is All You Need":
1. The sine and cosine of the positional encoding are concatenated not interweaved. Interweaving can be a simple future implementation. Apparently there isn't much of a different functionally, but it is worth investigating.
2. The Transformer model only contains one encoder layer and one decoder layer. Testing with various layer amounts may produce more desirable results.
