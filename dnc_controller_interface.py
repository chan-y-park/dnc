import tensorflow as tf

from utils import (
    oneplus,
    get_linear_outputs,
)

def build_interface(
    controller_outputs,
#    minibatch_size,
    num_read_heads,
    num_write_heads,
    width_memory_row,
    num_memory_row,
    variable_initializer,
):
#    B = minibatch_size
#    N = num_memory_row
    W = width_memory_row
    R = num_read_heads
    E = num_write_heads

    interface_dict = {}

    # k^{r, i}_t
    interface_dict['read_keys'] = get_linear_outputs(
        controller_outputs,
        outputs_shape=[R, W],
        initializer=variable_initializer(),
        name='read_keys',
    )
    # \hat{beta}^{r, i}_t
    interface_dict['read_strengths_pre_oneplus'] = get_linear_outputs(
        controller_outputs,
        shape=[R],
        initializer=variable_initializer(),
        name='read_strength_pre_oneplus',
    )
    interface_dict['read_strengths'] = oneplus(
        interface_dict['read_strengths_pre_oneplus'],
        name='read_strengths',
    )
    # k^{w}_t
    interface_dict['write_keys'] = get_linear_outputs(
        controller_outputs,
        outputs_shape=[E, W],
        initializer=variable_initializer(),
        name='write_keys',
    )
    # \hat{beta}^{w}_t
    interface_dict['write_strength_pre_oneplus'] = get_linear_outputs(
        controller_outputs,
        outputs_shape=[E],
        initializer=variable_initializer(),
        name='write_strength_pre_oneplus',
    )
    interface_dict['write_strength'] = oneplus(
        interface_dict['write_strength_pre_oneplus'],
        name='write_strength',
    )
    # \hat{e}_t
    interface_dict['erase_pre_sigmoid'] = get_linear_outputs(
        controller_outputs,
        outputs_shape=[E, W],
        initializer=variable_initializer(),
        name='erase_pre_sigmoid',
    )
    interface_dict['erase'] = tf.sigmoid(
        interface_dict['erase_pre_sigmoid'],
        name='erase',
    )
    # \nu_t
    interface_dict['write'] = get_linear_outputs(
        controller_outputs,
        outputs_shape=[E, W],
        initializer=variable_initializer(),
        name='write',
    )
    # f^i_t
    interface_dict['free_gates_pre_sigmoid'] = get_linear_outputs(
        controller_outputs,
        outputs_shape=[R],
        initializer=variable_initializer(),
        name='free_gates_pre_sigmoid',
    )
    interface_dict['free_gates'] = tf.sigmoid(
        interface_dict['free_gates_pre_sigmoid'],
        name='free_gates',
    )
    # g^a_t
    interface_dict['allocation_gates_pre_sigmoid'] = get_linear_outputs(
        controller_outputs,
        outputs_shape=[E],
        initializer=variable_initializer(),
        name='allocation_gates_pre_sigmoid',
    )
    interface_dict['allocation_gates'] = tf.sigmoid(
        interface_dict['allocation_gates_pre_sigmoid'],
        name='allocation_gates',
    )
    # g^w_t
    interface_dict['write_gates_pre_sigmoid'] = get_linear_outputs(
        controller_outputs,
        outputs_shape=[E],
        initializer=variable_initializer(),
        name='write_gates_pre_sigmoid',
    )
    interface_dict['write_gates'] = tf.sigmoid(
        interface_dict['write_gates_pre_sigmoid'],
        name='write_gates',
    )
    # pi^i_t
    num_read_modes  = 1 + 2 * E
    interface_dict['read_modes_pre_softmax'] = get_linear_outputs(
        controller_outputs,
        outputs_shape=[R, num_read_modes],
        initializer=variable_initializer(),
        name='read_modes_pre_softmax',
    )
    interface_dict['read_modes'] = tf.nn.softmax(
        interface_dict['read_modes_pre_softmax'],
        name='read_modes'
    )

    return interface_dict
