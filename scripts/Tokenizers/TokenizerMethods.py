from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow as tf
from Tokenizers.CustomTokenizer import CustomTokenizer

def create_tokenizers(train_example, input_vocab_path = "./input_vocab.txt", label_vocab_path = "./label_vocab.txt", tokenizer_output_path = "./ted_hrlr_translate_pt_en_converter"):
    train_input = train_example.map(lambda input, label: input) # Grab the text to be translated.
    train_label = train_example.map(lambda input, label: label) # Grab expected translation.

    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

    bert_tokenizer_params = dict(lower_case = True)
    bert_vocab_args = dict(
        # The desired size of the vocabulary.
        vocab_size = 8000,
        # Reserved tokens/words that must be included in the vocabulary.
        reserved_tokens = reserved_tokens,
        # Argments for text.BertTokenizer.
        bert_tokenizer_params = bert_tokenizer_params,
        # Arguments for wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn.
        learn_params={},
    )

    # Generates vocab for the text to be translated and translated text, respectively.
    input_vocab = bert_vocab.bert_vocab_from_dataset(train_input.batch(1000).prefetch(2), **bert_vocab_args)
    label_vocab = bert_vocab.bert_vocab_from_dataset(train_label.batch(1000).prefetch(2), **bert_vocab_args)

    # Saves vocabs to files.
    write_vocab_file(input_vocab_path, input_vocab)
    write_vocab_file(label_vocab_path, label_vocab)

    # Create tokenizers.
    tokenizers = tf.Module()
    tokenizers.input = CustomTokenizer(reserved_tokens, input_vocab_path)
    tokenizers.label = CustomTokenizer(reserved_tokens, label_vocab_path)

    tf.saved_model.save(tokenizers, tokenizer_output_path)

    return tokenizers


def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file = f)
