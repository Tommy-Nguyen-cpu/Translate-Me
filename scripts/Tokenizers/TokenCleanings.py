import tensorflow as tf
import re

def add_start_end(ragged, start, end):
    
    count = ragged.bounding_shape()[0] # Grab the outer dimension shape of ragged tensor.
    starts = tf.fill([count, 1], start) # Create tensor of shape "[count, 1]" filled with "[START]" tokens. There will be "count" inner tensors, and each tensor will contain 1 "[START]" token.
    ends = tf.fill([count, 1], end) # Same explanation as starts.

    return tf.concat([starts, ragged, ends], axis = 1) # Combines starts, ragged, and end into 1 tensor along axis 1 (probably last axis). In other words, combines elements in all three tensors into 1 tensor.

def cleanup_text(reserved_tokens, token_txt):
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"] # Drops reserved tokens except for [UNK].
    bad_tokens_re = "|".join(bad_tokens) # Concatenate "|" and bad_tokens into one string.

    bad_cells = tf.strings.regex_full_match(token_txt, bad_tokens_re) # Find all bad tokens in token_txt.
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells) # Keep good tokens. "~bad_cells" mean "not bad_cells".

    # Join them into a string.
    result = tf.strings.reduce_join(result, separator = " ", axis = -1) # Join along inner most axis.
    return result