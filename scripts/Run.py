import tensorflow as tf
import tensorflow_datasets as tfds
from DataPreprocessor import DataPreprocessor
from Transformer import Transformer
from CustomScheduler import CustomSchedule
from Translator import Translator
from ExportTranslator import ExportTranslator
from Tokenizers.TokenizerMethods import create_tokenizers


def load_tokenizers(path = "./ted_hrlr_translate_pt_en_converter"):
    return tf.saved_model.load(path)

def masked_loss(label, pred):
    mask = label != 0 # False at locations where the label is 0 and true everywhere else.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = "none") # Object that will compute the loss.
    loss = loss_object(label, pred) # Computes loss.

    mask = tf.cast(mask, dtype = loss.dtype) # Casts mask to be the same type as loss (either float or int).

    loss *= mask # Zero out all spots where the label is 0 and scale up all other locations.

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask) # Calculates the percentage of incorrectly classified tokens.

    return loss

def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis = 2) # Grabs the maximum prediction along axis 2.
    label = tf.cast(label, dtype = pred.dtype) # Cast label to be the same type as pred.
    match = label == pred # True where label and pred have the same value and false everywhere else.

    mask = label != 0 # False where label is 0 and true everywhere else.
    match = match & mask # Find all positions where the label and pred have the same value and the value is not 0.

    # Cast both to be floats/
    match = tf.cast(match, dtype = tf.float32)
    mask = tf.cast(mask, dtype = tf.float32)

    return tf.reduce_sum(match) / tf.reduce_sum(mask) # Calculates the percentage of correctly classified tokens.

def save_model(translator, dir = "translator"):
    tf.saved_model.save(translator, export_dir = dir)

def load_model(dir):
    return tf.saved_model.load(dir)

def load_dataset(dataset_name = "ted_hrlr_translate/pt_to_en"):
    examples, metadata = tfds.load(dataset_name, with_info = True, as_supervised = True)
    return examples['train'], examples['validation']

def print_translation(sentence):
    print(f"{sentence:15s}") # TODO: Once Gradio UI is implemented, change to display in UI.

def train_model(model_output_path = "./trained_model", dataset_name = "ted_hrlr_translate/pt_to_en", epochs = 20, num_layers = 4, d_model = 128, dff = 512, num_heads = 8, dropout_rate = 0.1):
    # Load the training dataset.
    train_ex, validate_ex = load_dataset(dataset_name = dataset_name)

    # Since we haven't trained a model for this yet, we will also have to create a new tokenizer from scratch.
    tokenizers = create_tokenizers(train_example = train_ex)

    preprocessor = DataPreprocessor(tokenizers)

    # Convert our datasets into batches.
    train_batches = preprocessor.make_batches(train_ex)
    validate_batches = preprocessor.make_batches(validate_ex)

    # Initialize transformer.
    transformer = Transformer(num_layers = num_layers, d_model = d_model, 
                              num_heads = num_heads, dff = dff, input_vocab_size = tokenizers.input.get_vocab_size().numpy(),
                              target_vocab_size = tokenizers.label.get_vocab_size().numpy(), dropout_rate = dropout_rate)
    
    # Initialize learning rate scheduler and optimizer.
    learning_schedule = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_schedule, beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9)

    # Compile & train our model.
    transformer.compile(optimizer = optimizer, loss = masked_loss, metrics = [masked_accuracy])
    transformer.fit(train_batches, epochs = epochs, validation_data = validate_batches)

    # Automate translation generation process.
    translator = Translator(tokenizers = tokenizers, transformer = transformer)

    # Wrap translator in another class that has a tf.function "__call__" method.
    translator = ExportTranslator(translator)

    # Save model.
    tf.saved_model.save(translator, model_output_path)

    return translator