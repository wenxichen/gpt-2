import numpy as np
import tensorflow as tf

HPARAMS_117M = {
  "n_vocab": 50257,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12
}

HPARAMS_355M = {
  "n_vocab": 50257,
  "n_ctx": 1024,
  "n_embd": 1024,
  "n_head": 16,
  "n_layer": 24
}

HPARAMS = {"117M": HPARAMS_117M,
           "355M": HPARAMS_355M}


def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly.
    x is a tensor."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

# def create_look_ahead_mask(size):
#     mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#     return mask  # (seq_len, seq_len)

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

def mask_attn_weights(w):
    # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    _, _, nd, ns = shape_list(w)
    b = attention_mask(nd, ns, dtype=w.dtype)
    b = tf.reshape(b, [1, 1, nd, ns])
    w = w*b - tf.cast(1e10, w.dtype)*(1-b)
    return w

def scaled_dot_product_attention(q, k, v):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, 
    i.e.: seq_len_k = seq_len_v.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)

    Returns:
        output, attention_weights
    """
    
    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)  

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    scaled_attention_logits = mask_attn_weights(scaled_attention_logits)  

    # softmax is normalized on the last axis (seq_len_k) 
    # so that the scores add up to 1.
    # (..., seq_len_q, seq_len_k)
    attention_weights = \
        tf.nn.softmax(scaled_attention_logits, axis=-1)

    # (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)  

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name):
        super(MultiHeadAttention, self).__init__(name=name)
        
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads

        w_init_1 = tf.random_normal_initializer(stddev=0.02)
        w_init_2 = tf.random_normal_initializer(stddev=0.02)
        self.dense1 = tf.keras.layers.Dense(d_model*3, kernel_initializer=w_init_1, name='c_attn')
        self.dense2 = tf.keras.layers.Dense(d_model, kernel_initializer=w_init_2, name='c_proj')
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is 
        (batch_size, num_heads, seq_len, depth)
        """
        n = self.num_heads
        *start, m = shape_list(x)
        x = tf.reshape(x, start + [n, m//n])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def merge_heads(self, x):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        *start, a, b = shape_list(x)
        return tf.reshape(x, start + [a*b])

    def call(self, X, past):
        # X: (batch, sequence, d_model)

        batch_size = tf.shape(X)[0]
        c = self.dense1(X)
        q, k, v = tf.split(c, 3, axis=2)
        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)  
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        # scaled_attention.shape == 
        #   (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == 
        #   (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = \
            scaled_dot_product_attention(q, k, v)

        # (batch_size, seq_len_q, d_model)
        concat_attention = self.merge_heads(scaled_attention)

        # (batch_size, seq_len_q, d_model)
        output = self.dense2(concat_attention)

        return output, present, attention_weights


# def point_wise_feed_forward_network(d_model, dff, name):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     return tf.keras.Sequential([
#         # (batch_size, seq_len, dff)
#         tf.keras.layers.Dense(dff, activation=gelu, kernel_initializer=w_init, name='c_fc'),
#         # (batch_size, seq_len, d_model)
#         tf.keras.layers.Dense(d_model, kernel_initializer=w_init, name='c_proj')  
#   ], name=name)


class PointWiseFF(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, name):
        super(PointWiseFF, self).__init__(name=name)
        w_init = tf.random_normal_initializer(stddev=0.02)
        # (batch_size, seq_len, dff)
        self.dense1 = tf.keras.layers.Dense(dff, activation=gelu, kernel_initializer=w_init, name='c_fc')
        # (batch_size, seq_len, d_model)
        self.dense2 = tf.keras.layers.Dense(d_model, kernel_initializer=w_init, name='c_proj')

    def call(self, X):
        return self.dense2(self.dense1(X))
        

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, name):
        super(DecoderLayer, self).__init__(name=name)
        self.mha = MultiHeadAttention(d_model, num_heads, name='attn')

        self.ffn = PointWiseFF(d_model, dff, name='mlp')

        self.layernorm1 = \
            tf.keras.layers.LayerNormalization(epsilon=1e-5, name='ln_1')
        self.layernorm2 = \
            tf.keras.layers.LayerNormalization(epsilon=1e-5, name='ln_2')

    def call(self, X, past):
        # X: (batch, sequence, d_model)
        X_norm = self.layernorm1(X)
        # attn: (batch_size, target_seq_len, d_model)
        # present: (batch_size, 2, num_heads, seq_len_v, depth)
        # where depth = d_model // num_heads
        # attn_weihgts_block: (batch_size, num_heads, seq_len_q, seq_len_k)
        attn, present, attn_weights_block = self.mha(
            X_norm, past
        )
        X_res_1 = attn + X
        X_res_1_norm = self.layernorm2(X_res_1)
        # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(X_res_1_norm)
        out = ffn_output + X_res_1
        return out, present, attn_weights_block


class Decoder(tf.keras.layers.Layer):
    def __init__(self, hparams, name, **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.num_layers = hparams['n_layer']
        self.d_model = hparams['n_embd']
        self.num_heads = hparams['n_head']
        self.dec_layers = [
            DecoderLayer(self.d_model, self.num_heads, self.d_model*4, name=('h' + str(i)))
            for i in range(self.num_layers)
        ]
        self.layernorm = \
            tf.keras.layers.LayerNormalization(epsilon=1e-5, name='ln_f')

    def call(self, X, past):
        # X: (batch, sequence, d_model)
        attention_weights = []
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None \
                else [None] * self.num_layers
        assert len(pasts) == self.num_layers
        for i in range(self.num_layers):
            # X: (batch_size, target_seq_len, d_model)
            # present: (batch_size, 2, num_heads, seq_len_v, depth)
            X, present, attn_weights_block = \
                self.dec_layers[i](X, pasts[i])
            presents.append(present)
            attention_weights.append(attn_weights_block)
        # (batch_size, num_layers, 2, num_heads, seq_len_v, depth)
        combined_present = tf.stack(presents, axis=1)
        out = self.layernorm(X)
        return out, combined_present, attention_weights


def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


class SharedEmbeddings(tf.keras.layers.Layer):
    """Perform shared embedding. Code adapt from https://github.com/huggingface/transformers/blob/dc3f6758cfe84a29fa87e9ce05b9b45e10cdb155/src/transformers/modeling_tf_utils.py#L1382"""
    def __init__(self, vocab_size, hidden_size, name='gpt2_tf2'):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.weight = self.add_weight(
            "weight",
            shape=[self.vocab_size, self.hidden_size], 
            initializer="random_normal",
            trainable=True
        )
        # super().build(input_shape)

    def _embedding(self, input_ids):
        """Applies embedding based on inputs tensor."""
        return tf.gather(self.weight, input_ids)

    def _linear(self, inputs):
        """
        Computes logits by running inputs through a linear layer.

        Args:
            inputs: A float32 tensor with shape [..., hidden_size]

        Returns:
            float32 tensor with shape [..., vocab_size].
        """
        first_dims = shape_list(inputs)[:-1]
        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.weight, transpose_b=True)

        return tf.reshape(logits, first_dims + [self.vocab_size])

    def call(self, inputs, mode):
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))


class GPT2(tf.keras.Model):
    def __init__(self, hparams, name='gpt2_tf2', train_wte_weight_only=False, **kwargs):
        super(GPT2, self).__init__(name=name, **kwargs)
        self.hparams = hparams
        self.wte = SharedEmbeddings(
            hparams['n_vocab'], hparams['n_embd'], name="wte"
        )
        self.wpe = tf.keras.layers.Embedding(
            hparams['n_ctx'],
            hparams['n_embd'],
            embeddings_initializer="random_normal",
            name="wpe",
            trainable=(not train_wte_weight_only)
        )
        self.decoder = Decoder(hparams, name='decoder', trainable=(not train_wte_weight_only))


    def call(self, X, past):
        # X: (batch_size, sequence)
        batch, sequence = shape_list(X)
        # (batch_size, sequence, d_model)
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = self.wte(X, mode="embedding") + self.wpe(positions_for(X, past_length))
        # out: (batch_size, target_seq_len, d_model)
        # present: (batch_size, num_layers, 2, num_heads, seq_len_v, depth)
        # attn_weights: list of (batch_size, num_heads, seq_len_q, seq_len_k) with length num_layers
        out, present, attn_weights = self.decoder(h, past)
        final_output = self.wte(out, mode="linear")
        return final_output, present, attn_weights

def init_GPT2_model_vars(gpt2):
    X = tf.convert_to_tensor(np.array([[35, 789], [98, 69]]))
    logits, presents, _ = gpt2(X, None)

def create_GPT2_model(hparams, name='gpt2_tf2', train_wte_weight_only=False):
    """Create and initialize the model variables."""
    gpt2 = GPT2(hparams, name=name, train_wte_weight_only=train_wte_weight_only)
    init_GPT2_model_vars(gpt2)
    return gpt2