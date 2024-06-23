
import tensorflow as tf

class DataPreprocessor():
    def __init__(self, tokenizers, max_tokens = 128):
        self.tokenizers = tokenizers
        self.max_tokens = max_tokens

    def prepare_batch(self, input_data, label_data):
        input_data = self.tokenizers.input.tokenize(input_data) # Output is RaggedTensor. Converts string to tokens.
        input_data = input_data[:, :self.max_tokens] # Trims to max_tokens.
        input_data = input_data.to_tensor() # Converts to 0-padded dense tensor.

        label_data = self.tokenizers.label.tokenize(label_data)
        label_data = label_data[:, :(self.max_tokens+1)] # Shifts the label data by 1 so it also predicts the next token.

        label_inputs = label_data[:, :-1].to_tensor() # Drop the [END] tokens.
        label_labels = label_data[:, 1:].to_tensor() # Drop the [START] tokens.

        return (input_data, label_inputs), label_labels

    def make_batches(self, ds, buffer_size = 6000, batch_size = 64):
        return (ds
                .shuffle(buffer_size) # Shuffles the dataset using "buffer_size" number of elements each time.
                .batch(batch_size) # Batch data into batches with "batch_size" number of elements. Tokenizers work more efficiently on larger batches.
                .map(self.prepare_batch, tf.data.AUTOTUNE) # Runs "prepare_batch" on each batch of data in REAL TIME (tf.data.AUTOTUNE ensures that the operation runs in real time).
                .prefetch(buffer_size = tf.data.AUTOTUNE)) # Fetches the data during training instead of having all of the data in memory at all time (which can be memory intensive).
