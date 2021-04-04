import tensorflow as tf

import model_tf2


def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.math.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.math.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    """Nucleus sampling"""
    if p == 0:
        return logits

    # It seems 0 top_p will just return the highest.
    batch = logits.shape[0]
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.math.cumsum(
        tf.nn.softmax(sorted_logits, axis=-1), axis=-1
    )
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(
            tf.math.reduce_sum(
                tf.cast(cumulative_probs <= p, tf.int32), 
                axis=-1
            ) - 1, 
            0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(*, gpt2_model, length, start_token=None, 
                    batch_size=None, context=None, temperature=1, 
                    top_k=0, top_p=1):
    if start_token is None:
        assert context is not None, \
        'Specify exactly one of start_token and context!'
    else:
        assert context is None, \
        'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)
        
    output = tf.cast(tf.identity(context), tf.int32)
    past = None
    
    for _ in range(length):
        mask = model_tf2.create_look_ahead_mask(context.shape[1])
        out, present, att_w = gpt2_model(context, past, mask)
        next_token_logits = \
            out[:,-1,:] / tf.cast(temperature, tf.float32)
        next_token_logits = top_k_logits(next_token_logits, k=top_k)
        next_token_logits = top_p_logits(next_token_logits, p=top_p)
        samples = tf.random.categorical(
            next_token_logits, num_samples=1, dtype=tf.int32
        )

        past = present if past is None \
            else tf.concat([past, present], axis=-2)
        context = samples
        output = tf.concat([output, samples], axis=1)
        
    return output
