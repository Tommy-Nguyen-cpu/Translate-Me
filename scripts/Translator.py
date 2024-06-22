import tensorflow as tf

class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer
    
    def __call__(self, sentence, max_length = 500):
        # The input sentence will need to include the '[START]' and '[END]' tokens.
        assert isinstance(sentence, tf.Tensor) # Verify that the sentence is a tensor.
        if len(sentence.shape) == 0: # If the sentence tensor is only 1D, add another dimension.
            sentence = sentence[tf.newaxis]
        
        sentence = self.tokenizers.input.tokenize(sentence).to_tensor() # Tokenize sentence and convert to tensor.
        encoder_input = sentence

        # Our output should also contain the '[START]' token.
        start_end = self.tokenizers.label.tokenize([''])[0] # Grab the only Tensor within the nested tensor ([[...]]).
        start = start_end[0][tf.newaxis] # Grab the start token and make the token 2D.
        end = start_end[1][tf.newaxis] # Grab the end token and make the token 2D.

        # We are going to be using "tf.TensorArray" instead of lists because our loop will only be registered as dynamic by "tf.function" if we use "tf.TensorArray".
        output_array = tf.TensorArray(dtype = tf.int64, size = 0, dynamic_size = True) # Initialize empty array that can add elements.
        output_array.write(0, start) # Write the start token as our first element in the output.

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack()) # ".stack" converts TensorArray to Tensor, and ".transpose" rotates the tensor.
            predictions = self.transformer([encoder_input, output], training = False) # Inference without training

            # Grab the last set of predictions from the "sequence_length" dimension.
            predictions = predictions[:, -1:, :] # Shape: (batch_size, 1, vocab_size) <- 1 was originally "sequence_length". We grabbed one so now its 1.
            predicted_id = tf.argmax(predictions, axis = -1) # Grab the highest value in the entire Tensor.

            # Add the predicted_id to our output, which was fed to our decoder as input.
            output_array.write(i+1, predicted_id)

            # If the predicted token is the end token, then we reached the end of the sentence and should stop generating.
            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack()) # Converts to Tensor and rotates Tensor.

        # The output is '(1, tokens)'.
        text = self.tokenizers.label.detokenize(output)[0] # Shape: "()".
        tokens = self.tokenizers.label.lookup(output)[0] # Look up tokens corresponding to output.

        # 'tf.function' prevents us from using the attention weights calculated in the last iteration of our loop.
        # We will recalculate them here.
        self.transformer([encoder_input, output[:, :-1]], training = False)
        attention_weights = self.transformer.decoder.last_attn_score

        return text, tokens, attention_weights