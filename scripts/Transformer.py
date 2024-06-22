from Layers.Encoder import Encoder
from Layers.Decoder import Decoder
import tensorflow as tf

class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate = 0.1):
        super().__init__()

        self.encoder = Encoder(num_layers = num_layers, d_model = d_model, num_heads = num_heads, dff = dff, vocab_size = input_vocab_size, dropout_rate = dropout_rate) # Encodes input.
        self.decoder = Decoder(num_layers = num_layers, d_model = d_model, num_heads = num_heads, dff = dff, vocab_size = target_vocab_size, dropout_rate = dropout_rate) # Decodes inputs and predicts next token.

        # Converts output of decoder into probability of next token occurring.
        self.final_dense_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inputs):
        # In order to use the ".fit" method, we must pass the x and context as 1 argument.
        x, context = inputs

        context = self.encoder(context) # # (batch_size, sequence_length, d_model)

        x = self.decoder(x, context) # (batch_size, target_sequence_length, d_model)

        # Final linear layer output.
        logits = self.final_dense_layer(x) # (batch_size, target_sequence_length, target_vocab_size)

        try:
            # Drop the mask so it doesn't scale the losses/metrics.
            del logits._keras_mask
        except AttributeError:
            pass

        # Returns the final output and attention weights.
        return logits