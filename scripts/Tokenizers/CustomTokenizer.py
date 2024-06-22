import tensorflow as tf
import tensorflow_text
import pathlib
from Tokenizers.TokenCleanings import add_start_end, cleanup_text

class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = tensorflow_text.BertTokenizer(vocab_path, lower_case = True) # Initialize BertTokenizer and performs preprocessing on text when called.
        self.reserved_tokens = reserved_tokens
        # Find all instances of start token and grab the indices of 1 of those instances/
        self.START = tf.argmax(tf.constant(reserved_tokens) == "[START]") # "tf.constant" can't be modified. "tf.argmax" will then grab indices of largest value.
        self.END = tf.argmax(tf.constant(reserved_tokens) == "[END]") # Grab last occurrence of end token.

        self.vocab_path = tf.saved_model.Asset(vocab_path) # Include the file at the path "vocab_path" in our saved model.

        vocab = pathlib.Path(vocab_path).read_text().splitlines() # Reads text and splits based on newline.
        self.vocab = tf.Variable(vocab) # Vocabulary can change.

        # Create signature for export

        # Include tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(tf.TensorSpec(shape = [None], dtype = tf.string))

        # Include detokenize and lookup signatures for:
            # * 'Tensors' with shape "[tokens]" and "[batch, tokens]"
            # * 'RaggedTensor' with shape "[batch, tokens]"
        self.detokenize.get_concrete_function(tf.TensorSpec(shape = [None, None], dtype = tf.int64))
        self.detokenize.get_concrete_function(tf.RaggedTensorSpec(shape = [None, None], dtype = tf.int64))

        # These "get_*" methods take no arguments.
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings) # Tokenize text.

        enc = enc.merge_dims(-2, -1) # Merge word and wordpiece axis.
        enc = add_start_end(enc, self.START, self.END) # Add the start and end tokens.
        
        return enc
    
    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized) # Converts tokens to words.
        return cleanup_text(self.reserved_tokens, words) # Remove reserved tokens from output and merge words back into strings.
    
    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)
    
    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]
    
    @tf.function
    def get_vocab_path(self):
        return self.vocab_path
    
    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self.reserved_tokens) # Returns immutable instance of reserved tokens.