from Layers.AttentionLayers.BaseAttention import BaseAttention

class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_score = self.mha(query = x, key = context, value = context, return_attention_scores=True) # Computes weighted average and attention score

        self.last_attn_score = attn_score

        x = self.add([x, attn_output]) # Adds the attention output of our encoder to the output of our decoder.
        x = self.layernorm(x) # Normalize data.

        return x