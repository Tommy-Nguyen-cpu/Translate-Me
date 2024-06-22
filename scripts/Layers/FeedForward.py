import tensorflow as tf

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate = 0.1):
        super().__init__()
        # "dff" is the feed forward upward projection size, basically the dimensions of the output.
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="relu"), # "dff" neurons, resulting in "dff" outputs.
            tf.keras.layers.Dense(d_model), # "d_model" neurons.
            tf.keras.layers.Dropout(dropout_rate) # Ensures neurons are self sufficient and do not rely to much on other neurons.
        ])

        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        # "self.seq" transforms data into something we understand.
        x = self.add([x, self.seq(x)]) # Add the input and values predicted from the feed forward network.
        x = self.layernorm(x) # Normalize data

        return x