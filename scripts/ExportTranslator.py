import tensorflow as tf

class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator
    
    @tf.function(input_signature=[tf.TensorSpec(shape = [], dtype = tf.string), tf.TensorSpec(shape = [], dtype = tf.int64)])
    def __call__(self, sentence, max_tokens):
        (result, tokens, attention_weights) = self.translator(sentence, max_length = max_tokens)
        return result