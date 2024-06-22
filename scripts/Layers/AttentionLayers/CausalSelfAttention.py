from Layers.AttentionLayers.BaseAttention import BaseAttention

class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query = x, key = x, value = x, use_causal_mask = True) # Predict tokens based solely on previous tokens generated, NOT on the entire sequence of tokens/sentence inputted.

        x = self.add([x, attn_output]) # Add attention output to x.
        x = self.layernorm(x) # Normalize data.

        return x