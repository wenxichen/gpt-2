import numpy as np
import tensorflow as tf

HPARAMS = {
  "n_vocab": 50257,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12
}

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly.
    x is a tensor."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
    

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, 
    i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type
    (padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """
    
    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)  

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) 
    # so that the scores add up to 1.
    # (..., seq_len_q, seq_len_k)
    attention_weights = \
        tf.nn.softmax(scaled_attention_logits, axis=-1)

    # (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)  

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads

        w_init = tf.random_normal_initializer(stddev=0.02)
        self.dense1 = tf.keras.layers.Dense(d_model*3, kernel_initializer=w_init)
        self.dense2 = tf.keras.layers.Dense(d_model, kernel_initializer=w_init)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is 
        (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.depth)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, X, past, mask):
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
            scaled_dot_product_attention(q, k, v, mask)
        concat_attention = tf.reshape(
            scaled_attention,                       
            (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)
        # (batch_size, seq_len_q, d_model)
        output = self.dense2(concat_attention)

        return output, present, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    w_init = tf.random_normal_initializer(stddev=0.02)
    return tf.keras.Sequential([
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(dff, activation=gelu, kernel_initializer=w_init),  
        # (batch_size, seq_len, d_model)
        tf.keras.layers.Dense(d_model, kernel_initializer=w_init)  
  ])


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(DecoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = \
            tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.layernorm2 = \
            tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, X, past, mask):
        # X: (batch, sequence, d_model)
        X_norm = self.layernorm1(X)
        # attn: (batch_size, target_seq_len, d_model)
        # present: (batch_size, 2, num_heads, seq_len_v, depth)
        # where depth = d_model // num_heads
        # attn_weihgts_block: (batch_size, num_heads, seq_len_q, seq_len_k)
        attn, present, attn_weights_block = self.mha(
            X_norm, past, mask
        )
        X_res_1 = attn + X
        X_res_1_norm = self.layernorm2(X_res_1)
        # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(X_res_1_norm)
        out = ffn_output + X_res_1
        return out, present, attn_weights_block


class Decoder(tf.keras.layers.Layer):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.num_layers = hparams['n_layer']
        self.d_model = hparams['n_embd']
        self.num_heads = hparams['n_head']
        self.dec_layers = [
            DecoderLayer(self.d_model, self.num_heads, self.d_model*4) 
            for _ in range(self.num_layers)
        ]
        self.layernorm = \
            tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, X, past, mask):
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
                self.dec_layers[i](X, pasts[i], mask)
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


class Embedding(tf.keras.layers.Layer):
    def __init__(self, wpe, wte):
        super(Embedding, self).__init__()
        self.wpe = wpe
        self.wte = wte

    def call(self, X, past):
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(self.wte, X) + tf.gather(self.wpe, positions_for(X, past_length))
        return h


class LookUp(tf.keras.layers.Layer):
    def __init__(self, wte):
        super(LookUp, self).__init__()
        self.wte = wte
    
    def call(self, X):
        batch_size, sequence, _ = shape_list(X)
        X_flat = tf.reshape(X, [batch_size*sequence, -1])
        out = tf.matmul(X_flat, self.wte, transpose_b=True)
        out = tf.reshape(out, [batch_size, sequence, -1])
        return out


class GPT2(tf.keras.Model):
    def __init__(self, hparams):
        super(GPT2, self).__init__()
        self.hparams = hparams
        wpe_init = tf.random_normal_initializer(
            mean=0.0, stddev=0.01, seed=None
        )
        wte_init = tf.random_normal_initializer(
            mean=0.0, stddev=0.02, seed=None
        )
        self.wpe = tf.Variable(wpe_init(shape=[hparams['n_ctx'],hparams['n_embd']]))
        self.wte = tf.Variable(wte_init(shape=[hparams['n_vocab'],hparams['n_embd']]))
        self.embedding = Embedding(self.wpe, self.wte)
        self.decoder = Decoder(hparams)
        self.final_layer = LookUp(self.wte)


    def call(self, X, past, mask):
        # X: (batch_size, sequence)
        batch, sequence = shape_list(X)
        h = self.embedding(X, past) # (batch_size, sequence, d_model)
        # out: (batch_size, target_seq_len, d_model)
        # present: (batch_size, num_layers, 2, num_heads, seq_len_v, depth)
        # attn_weights: list of (batch_size, num_heads, seq_len_q, seq_len_k) with length num_layers
        out, present, attn_weights = self.decoder(h, past, mask)
        final_output = self.final_layer(out)
        return final_output, present, attn_weights