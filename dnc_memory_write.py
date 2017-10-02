import tensorflow as tf

from utils import get_content_weightings


def get_usages(
    prev_read_weightings,
    prev_write_weightings,
    prev_usages,
    new_dnc_state,
):
    # prev_read_weightings: [minibatch_size, R, N]
    # free_gates: [minibatch_size, R]
    # TODO: Maybe expand_dims not needed thanks to broadcasting.
    free_gates = tf.expand_dims(new_dnc_state['free_gates'], -1)

    # memory retention psi_t: [minibatch_size, N]
    memory_retentions = tf.reduce_prod(
        (1 - free_gates * prev_read_weightings),
        axis=[1]
    )

    # usage u_t: [-1, N]
    usages = (
        prev_usages + prev_write_weightings
        - prev_usages * prev_write_weightings
    ) * memory_retentions

    return usages


def get_allocation_weightings(
    usages,
):
    # free list phi_t: [-1, N]
    _usages = EPSILON + (1 - EPSILON) * usages
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
    return allocation_weightings


def get_write_weighting(
    allocation_gates,
    write_gates,
    allocation_weightings,
    write_content_weightings,
):
    """
    write_weightings: [-1, E, N], w_{t}^{w} in the paper.
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
    t,
):
    # TODO: get shape params using interface_dict.

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
    allocation_weightings = get_allocation_weightings(usages)

    # c_{t}^{w}
    write_content_weightings = get_content_weightings(
        interface_dict['write_keys'],
        memory,
        interface_dict['write_strengths'],
        name='write_content_weightings',
    )

    # w_{t}^{w}
    write_weightings = get_write_weightings(
        interface_dict['allocation_gates'],
        interface_dict['write_gates'],
        allocation_weightings,
        write_content_weightings,
    )

    erase_vectors = interface_dict['erase'] 
    write_vectors = interface_dict['write']

    # TODO: M_t = M_{t-1} \circ (1 - w^w_t e^T_t) + w^w_t \nu_^T_t 

    return (
        usages,
        write_weightings,
        memory,
    )
