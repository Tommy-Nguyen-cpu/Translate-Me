import tensorflow as tf
import numpy as np

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) # Converts input into indices (embedding vector). "mask_zero" determines whether part of embedding vector can be ignored because its just padding values.
        self.pos_encoding = self.positional_encoding(length = 2048, depth = d_model) # Compute positional encoding vector.


    def positional_encoding(self, length, depth):
        depth /= 2 # Splits length into 2, one for cosine and one for sine

        positions = np.arange(length)[:, np.newaxis] # Each element will be a numpy array.
        depths = np.arange(depth)[np.newaxis, :] / depth # 2D array with 1 inner array containing all of the elements. Divides each depth by the total depth.

        frequency = 1/ (10000 ** depths) # Calculates how long it should take to complete one cycle. 10,000 is to ensure we have enough unique positional encodings.
        radians = positions * frequency # Calculates position within cycle.

        pos_encodings = np.concatenate([np.sin(radians), np.cos(radians)], axis=-1) # Merge numpy arrays by inner most elements (sine, cosine, sine, cosine, sine...)

        return tf.cast(pos_encodings, dtype=tf.float32) # Converts values to floats.

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs) # Returns a mask that tells layers whether some data is missing. If they are missing, then we should skip over those missing data and not compute for them.
    
    def call(self, x):
        length = tf.shape(x)[1] # Counts all words in the input.
        x = self.embedding(x) # Converts x to embedding vector.

        # Normalize variance of the embedding layer. Keeps positional encoding and embedding layer to be of same variance/scale.
        x *= tf.sqrt(tf.cast(self.d_model, dtype=tf.float32))

        x = x + self.pos_encoding[tf.newaxis, :length, :] # Apply positional encoding to x. Grab the first "length" rows, and adds an extra dimension to ensure positional encoding matches shape of x.

        return x