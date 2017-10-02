import tensorflow as tf

from utils import get_content_weightings


def get_temporal_memory_linkage(
    write_weighting,
    prev_precedence_weighting,
    prev_temporal_memory_linkage,
):
    # TODO Implement the sparse version.
    precedence_weighting = (
        1 - tf.reduce_sum(write_weighting, axis=2, keep_dims=True)
    ) * prev_precedence_weighting + write_weighting

    w_i = tf.expand_dims(write_weighting, axis=3)
    w_j = tf.expand_dims(write_weighting, axis=2)
    p_j = tf.expand_dims(prev_precedence_weighting, axis=2)
    L_ij = (1 - w_i - w_j) * prev_temporal_memory_linkage + w_i * p_j
    temporal_linkage = tf.matrix_set_diag(
        input=L_ij,
        diagonal=tf.zeros(write_weighting.shape.as_list(), dtype=tf.float32),
        name='temporal_linkage',
    )

    return temporal_linkage


def get_directional_read_weightings(
    temporal_memory_linkage,
    prev_read_weightings,
    forward=True,
):
    B, E, N, _ = temporal_memory_linkage.shape.as_list()
    # expanded_prev_w_r: [B, E, R, N]
    # the following is equivalent to 
    # tf.expand_dims(prev_read_weightings, axis=1) when E = 1.
    expanded_prev_w_r = tf.stack(
        [prev_read_weightings] * E,
        axis=1,
    )
    directional_read_weightings = tf.matmul(
        expanded_prev_w_r,
        temporal_memory_linkage,
        transpose_b=forward,
    )
    # returning tensor shape: [B, R, E, N]
    return tf.transpose(
        directional_read_weightings,
        [0, 2, 1, 3],
    )


def get_read_weightings(
    read_modes,
    backward_weightings,
    content_weightings,
    forward_weightings,
):
    B, R, E, N = backward_weightings.shape.as_list()
    # read_modes, \pi_{t}^{i}: [B, R, (1+2*E)]
    backward_modes, forward_modes, content_mode = tf.split(
        read_modes,
        [E, E, 1],
        axis=2,
    )

    # read_weightings, w_{t}^{i}: [B, R, N].
    read_weightings = tf.add_n(
        [
            tf.reduce_sum(
                tf.expand_dims(modes, 3) * weightings,
                axis=2,
            )
            for modes, weightings in (
                (backward_modes, backward_weightings), 
                (content_mode, content_weightings), 
                (forward_modes, forward_weightings), 
            )
        ]
    )

    return read_weightings


def read_memory(
    prev_precedence_weighting,
    prev_temporal_memory_linkage,
    prev_read_weightings,
    write_weightings,
    memory,
    interface_dict,
    t,
):
    # TODO: get shape params using interface_dict.

    # temporal memory link matrix, L_t.
    if t == 0:
        precedence_weighting = tf.zeros(
            shape=[B, E, N], 
            dtype=tf.float32,
        )
        temporal_memory_linkage = tf.zeros(
            shape=[B, E, N, N], 
            dtype=tf.float32,
        )
        # f_{t}^{i}: [B, R, E, N]
        read_forward_weighting = tf.zeros(
            shape=[B, R, E, N], 
            dtype=tf.float32,
        )
        # b_{t}^{i}: [B, R, E, N]
        read_backward_weighting = tf.zeros(
            shape=[B, R, E, N], 
            dtype=tf.float32,
        )
    else:
        temporal_memory_linkage = get_temporal_memory_linkage(
            write_weightings,
            prev_precedence_weighting,
            prev_temporal_memory_linkage,
        )
        # f_{t}^{i}: [B, R, E, N]
        read_forward_weighting = get_directional_read_weightings(
            temporal_memory_linkage,
            prev_read_weightings,
            forward=True,
        )
        # b_{t}^{i}: [B, R, E, N]
        read_backward_weighting = get_directional_read_weightings(
            temporal_memory_linkage,
            prev_read_weightings,
            forward=False,
        )

    # c_{t}^{i}: [B, R, E, N]
    read_content_weightings = get_content_weightings(
        interface_dict['read_keys'],
        memory,
        interface_dict['read_strengths'],
        name='read_content_weightings',
    )
    # read weightings w_{t}^{r, i}: [B, R, N].
    read_weightings = get_read_weightings(
        interface_dict['read_modes'],
        read_backward_weightings,
        read_content_weightings,
        read_forward_weightings,
    )

    # read vector r_{t}^{i}: [B, R, W].
    read_vectors = tf.matmul(
        memory,
        read_weightings,
    )

    return (
        precedence_weighting,
        temporal_memory_linkage,
        read_weightings,
        read_vectors,
    )
