from Layers.AttentionLayers.GlobalSelfAttention import GlobalSelfAttention
from Layers.FeedForward import FeedForward
import tensorflow as tf

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate = 0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(num_heads = num_heads, key_dim = d_model, dropout = dropout_rate) # Propogate information of every token across the entire sequence/sentence/
        self.ffn = FeedForward(d_model = d_model, dff = dff) # Convert attention output into something we can understand.

    def call(self, x):
        x = self.self_attention(x) # Produce vector that incorporates relationships of all tokens across all tokens in sequence.
        x = self.ffn(x) # Transforms data into proper representation (something we would understand).

        return x