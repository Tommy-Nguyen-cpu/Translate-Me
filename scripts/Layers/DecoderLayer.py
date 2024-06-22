from Layers.AttentionLayers.CausalSelfAttention import CausalSelfAttention
from Layers.AttentionLayers.CrossAttention import CrossAttention
from Layers.FeedForward import FeedForward
import tensorflow as tf

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate = 0.1):
        super(DecoderLayer, self).__init__()

        self.causal_attn_layer = CausalSelfAttention(num_heads = num_heads, key_dim = d_model, dropout = dropout_rate) # Propogates information from previous tokens onto the current token. Current token will be generated using information gathered from the previous generated tokens.
        self.cross_attn_layer = CrossAttention(num_heads = num_heads, key_dim = d_model, dropout = dropout_rate) # Gathers output from Encoder layer and adds it to the decoder output.
        self.ffn = FeedForward(d_model, dff) # Process information and transforms it into something we understand.

    def call(self, x, context):
        x = self.causal_attn_layer(x=x) # Gathers information from past tokens to be used to generate the current token.
        x = self.cross_attn_layer(x=x, context=context) # Combines decoder and encoder outputs.

        self.last_attn_score = self.cross_attn_layer.last_attn_score

        x = self.ffn(x) # Shape: (batch_size, sequence_length, d_model)

        return x