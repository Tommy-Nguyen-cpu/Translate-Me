import tensorflow as tf
from Layers.EncoderLayer import EncoderLayer
from Layers.PositionalEncoding import PositionalEmbedding

class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate = 0.1):
        """ Initializes the Encoder.
        num_layers = Defines the number of encoder layers.
        d_model = Determines the dimension of the input to the multiheadattention layer (also used in positional encoding).
        dff = The feed forward upward projection size; determines the shape of the output after passing through first layer in the feed forward network.
        vocab_size = Number of subwords in vocabulary.
        dropout_rate = rate at which a neuron may be turned off to allow for more efficient training.
        """

        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size = vocab_size, d_model = d_model) # Incorporates information about position of tokens.

        # Initializes "num_layers" encoder layers. These layers will perform the operation of learning patterns from the entire sequence of tokens.
        self.enc_layers = [EncoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout_rate = dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x):
        # 'x' is the token IDs with the shape: (batch_size, sequence_length)
        x = self.pos_embedding(x) # (batch_size, sequence_length, d_model)

        x = self.dropout(x) # Allows each neuron to learn independently.

        # Have each encoding layer process the input.
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        
        return x