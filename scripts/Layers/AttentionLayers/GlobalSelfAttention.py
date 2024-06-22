from Layers.AttentionLayers.BaseAttention import BaseAttention

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query = x, key = x, value = x) # Produce attention output.

        x = self.add([x, attn_output]) # Combines output of attention layer with the input.
        x = self.layernorm(x) # Normalize data.

        return x