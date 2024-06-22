from Layers.PositionalEncoding import PositionalEmbedding
from Layers.DecoderLayer import DecoderLayer
import tensorflow as tf

class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate = 0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size = vocab_size, d_model = d_model) # Incorporates positional information for tokens.
        self.dropout = tf.keras.layers.Dropout(dropout_rate) # Neurons will learn independently without relying too much on neighboring neurons.

        self.dec_layers = [DecoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout_rate = dropout_rate) for _ in range(self.num_layers)]
    
    def call(self, x, context):
        x = self.pos_embedding(x) # Apply positional information to query.

        x = self.dropout(x) # Produce output with dropout.

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context) # Combines encoder and decoder output and predict based on the combined output.
        
        self.last_attn_score = self.dec_layers[-1].last_attn_score

        return x # Shape: (batch_size, target_sequence_length, d_model)