import tensorflow as tf

from utils import (
    batch_invert_permutation,
    batch_gather,
    get_content_weightings,
)


def get_usages(
    prev_read_weightings,
    prev_write_weightings,
    prev_usages,
    new_dnc_state,
):
    # XXX Calculation of usage is not differentiable
    # with respect to write weights.
    prev_write_weightings = tf.stop_gradient(prev_write_weightings)

    # prev_read_weightings: [B, R, N]
    # free_gates: [B, R] -> [B, R, 1]
    free_gates = tf.expand_dims(new_dnc_state['free_gates'], axis=2)

    # memory retention psi_t: [B, N]
    memory_retentions = tf.reduce_prod(
        (1 - free_gates * prev_read_weightings),
        axis=[1],
    )

    # prev_write_weightings: [B, E, N] -> [B, N]
    prev_write_weightings = 1 - tf.reduce_prod(
        1 - prev_write_weightings, axis=[1],
    )
    # usage u_t: [B, N]
    usages = (
        prev_usages + prev_write_weightings
        - prev_usages * prev_write_weightings
    ) * memory_retentions

    return usages


def get_allocation_weightings(
    usages,
    epsilon,
):
    # free list phi_t: [B, N]
    _usages = epsilon + (1 - epsilon) * usages
    non_usages = 1 - _usages
    sorted_non_usages, sorted_non_usage_indices = tf.nn.top_k(
        non_usages,
        k=non_usages.shape.as_list()[-1],
    )
    sorted_usages = 1 - sorted_non_usages
    sorted_allocation_weightings = (
        sorted_non_usages
        * tf.cumprod(
            sorted_usages,
            axis=1,
            exclusive=True,
        )
    )
    sorted_usage_indices = batch_invert_permutation(sorted_non_usage_indices)
    allocation_weightings = batch_gather(
        sorted_allocation_weightings, 
        sorted_usage_indices,
    )
    # allocation_weightings: [B, E, N]
    return allocation_weightings


def get_write_weightings(
    allocation_gates,
    write_gates,
    allocation_weightings,
    write_content_weightings,
):
    """
    allocation_gates, write_gates: [B, E, 1].
    allocation_weightings: [B, E, N].
    write_content_weightings: [B, E, N].

    return write_weightings: [B, E, N], w_{t}^{w} in the paper.
    """

    write_weightings = (
        write_gates 
        * (allocation_gates * allocation_weightings
           + (1 - allocation_gates) * write_content_weightings)
    )
    return write_weightings


def write_memory(
    prev_read_weightings,
    prev_write_weightings,
    prev_usages,
    prev_memory,
    interface_dict,
    config,
    t,
):
    # prev_memory, memory: [B, N, W]

    B = config['minibatch_size']
    N = config['num_memory_rows']
    W = config['width_memory_row']
    R = config['num_read_heads']
    E = config['num_write_heads']
    epsilon = config['epsilon']


    # allocation_gates, write_gates: [B, E] -> [B, E, 1].
    allocation_gates = tf.expand_dims(
        interface_dict['allocation_gates'],
        axis=2,
    )
    write_gates = tf.expand_dims(
        interface_dict['write_gates'],
        axis=2,
    )

    # dynamic_memory_allocation, a_t 
    if t == 0:
        usages = tf.zeros(
            shape=[B, N],
            dtype=tf.float32,
        )
    else:
        usages = get_usages(
            prev_read_weightings,
            prev_write_weightings,
            prev_usages,
            interface_dict,
        )
    allocation_weightings = []
    for i_e in range(E):
        allocation_weightings.append(
            get_allocation_weightings(
                usages,
                epsilon,
            )
        )
        usages += ((1 - usages) * write_gates[:, i_e, :]
                  * allocation_weightings[i_e])
    # allocation_weightings: [B, N] * E -> [B, E, N].
    allocation_weightings = tf.stack(allocation_weightings, axis=1)

    # c_{t}^{w}: [B, E, N]
    write_content_weightings = get_content_weightings(
        interface_dict['write_keys'],
        prev_memory,
        interface_dict['write_strengths'],
        epsilon=epsilon,
        name='write_content_weightings',
    )

    # w_{t}^{w}: [B, E, N]
    write_weightings = get_write_weightings(
        allocation_gates,
        write_gates,
        allocation_weightings,
        write_content_weightings,
    )

    # e_{t}: [B, E, W]
    erase_vectors = interface_dict['erase'] 
    # \nu_{t}: [B, E, W]
    write_vectors = interface_dict['write']

    # M_t = M_{t-1} \circ (1 - w^w_t e^T_t) + w^w_t \nu_^T_t 
    memory = (
        prev_memory
        * (1 - tf.einsum('ben,bew->bnw', write_weightings, erase_vectors))
        + tf.einsum('ben,bew->bnw', write_weightings, write_vectors)
    )

    return (
        usages,
        write_weightings,
        memory,
    )
