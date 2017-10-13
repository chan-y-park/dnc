import numpy as np
import tensorflow as tf

def oneplus(tensor, name):
    return tf.add(
        tf.nn.softplus(tensor),
        1.0,
        name=name,
    )


# NOTE: From deepmind/dnc/utils.py
def batch_invert_permutation(permutations):
    """
    Returns batched `tf.invert_permutation` for every row in `permutations`.
    """
    with tf.name_scope('batch_invert_permutation', values=[permutations]):
        unpacked = tf.unstack(permutations)
        inverses = [tf.invert_permutation(permutation)
                    for permutation in unpacked]
        return tf.stack(inverses)


# NOTE: From deepmind/dnc/utils.py
# XXX: Can tf.gather() r1.3 replace this?
def batch_gather(values, indices):
    """
    Returns batched `tf.gather` for every row in the input.
    """
    with tf.name_scope('batch_gather', values=[values, indices]):
        unpacked = zip(tf.unstack(values), tf.unstack(indices))
        result = [tf.gather(value, index) for value, index in unpacked]
        return tf.stack(result)

def get_linear_outputs(
    inputs,
    outputs_shape,
    variable_initializer,
    name,
):
    # inputs: [B, controller_output_size]
    B, inputs_size = inputs.shape.as_list()
    linear_size = np.prod(outputs_shape)
    with tf.variable_scope(name):
        W = tf.get_variable(
            name='kernel',
            shape=[inputs_size, linear_size],
            initializer=variable_initializer,
        )
        b = tf.get_variable(
            name='bias',
            shape=[linear_size],
            initializer=tf.zeros_initializer(),
        )
        linear_outputs = tf.nn.xw_plus_b(
            inputs, W, b,
        )
        outputs = tf.reshape(
            linear_outputs,
            shape=([B] + outputs_shape),
            name=name,
        )
    return outputs


def get_content_weightings(
    keys,
    memory,
    strengths,
    epsilon,
    name=None,
):
    # keys: [-1, num_heads, W]
    # memory: [-1, N, W]
    # strengths: [-1, num_heads]

    # numerator: [-1, num_heads, N]
    numerator = tf.einsum(
        'bhw,bnw->bhn',
        keys,
        memory,
    )
    keys_norm = tf.sqrt(
        tf.norm(
            keys,
            axis=2,
        )
    )
    memory_norm = tf.sqrt(
        tf.norm(
            memory,
            axis=2,
        )
    )
    # denominator: [-1, num_heads, N]
    denominator = tf.einsum(
        'bh,bn->bhn',
        keys_norm,
        memory_norm,
    ) + epsilon

    # cosine_similarity: [-1, num_heads, N]
    cosine_similarity = numerator / denominator
    
    content_weighting_logits = tf.einsum(
        'bhn,bh->bhn',
        cosine_similarity,
        strengths,
    )
    content_weightings = tf.nn.softmax(
        content_weighting_logits,
        name=name,
    )
    return content_weightings
