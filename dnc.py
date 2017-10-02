import numpy as np
import tensorflow as tf

from utils import (
    get_linear_outputs,
)

EPSILON = 1e-6


#def build_state(
#    controller_outputs,
##    minibatch_size,
#    num_read_heads,
#    num_write_heads,
#    width_memory_row,
#    num_memory_row,
#    variable_initializer,
#):
##    B = minibatch_size
##    N = num_memory_row
#    W = width_memory_row
#    R = num_read_heads
#    E = num_write_heads
#
#    state_dict = {}
#    with tf.variable_scope('interface_parameters'):
#        # k^{r, i}_t
#        state_dict['read_keys'] = get_linear_outputs(
#            controller_outputs,
#            outputs_shape=[R, W],
#            initializer=variable_initializer(),
#            name='read_keys',
#        )
#        # \hat{beta}^{r, i}_t
#        state_dict['read_strengths_pre_oneplus'] = get_linear_outputs(
#            controller_outputs,
#            shape=[R],
#            initializer=variable_initializer(),
#            name='read_strength_pre_oneplus',
#        )
#        state_dict['read_strengths'] = oneplus(
#            state_dict['read_strengths_pre_oneplus'],
#            name='read_strengths',
#        )
#        # k^{w}_t
#        state_dict['write_keys'] = get_linear_outputs(
#            controller_outputs,
#            outputs_shape=[E, W],
#            initializer=variable_initializer(),
#            name='write_keys',
#        )
#        # \hat{beta}^{w}_t
#        state_dict['write_strength_pre_oneplus'] = get_linear_outputs(
#            controller_outputs,
#            outputs_shape=[E],
#            initializer=variable_initializer(),
#            name='write_strength_pre_oneplus',
#        )
#        state_dict['write_strength'] = oneplus(
#            state_dict['write_strength_pre_oneplus'],
#            name='write_strength',
#        )
#        # \hat{e}_t
#        state_dict['erase_pre_sigmoid'] = get_linear_outputs(
#            controller_outputs,
#            outputs_shape=[W],
#            initializer=variable_initializer(),
#            name='erase_pre_sigmoid',
#        )
#        state_dict['erase'] = tf.sigmoid(
#            state_dict['erase_pre_sigmoid'],
#            name='erase',
#        )
#        # \nu_t
#        state_dict['write'] = get_linear_outputs(
#            controller_outputs,
#            outputs_shape=[W],
#            initializer=variable_initializer(),
#            name='write',
#        )
#        # f^i_t
#        state_dict['free_gates_pre_sigmoid'] = get_linear_outputs(
#            controller_outputs,
#            outputs_shape=[R],
#            initializer=variable_initializer(),
#            name='free_gates_pre_sigmoid',
#        )
#        state_dict['free_gates'] = tf.sigmoid(
#            state_dict['free_gates_pre_sigmoid'],
#            name='free_gates',
#        )
#        # g^a_t
#        state_dict['allocation_gates_pre_sigmoid'] = get_linear_outputs(
#            controller_outputs,
#            outputs_shape=[E],
#            initializer=variable_initializer(),
#            name='allocation_gates_pre_sigmoid',
#        )
#        state_dict['allocation_gates'] = tf.sigmoid(
#            state_dict['allocation_gates_pre_sigmoid'],
#            name='allocation_gates',
#        )
#        # g^w_t
#        state_dict['write_gates_pre_sigmoid'] = get_linear_outputs(
#            controller_outputs,
#            outputs_shape=[E],
#            initializer=variable_initializer(),
#            name='write_gates_pre_sigmoid',
#        )
#        state_dict['write_gates'] = tf.sigmoid(
#            state_dict['write_gates_pre_sigmoid'],
#            name='write_gates',
#        )
#        # pi^i_t
#        num_read_modes  = 1 + 2 * E
#        state_dict['read_modes_pre_softmax'] = get_linear_outputs(
#            controller_outputs,
#            outputs_shape=[R, num_read_modes],
#            initializer=variable_initializer(),
#            name='read_modes_pre_softmax',
#        )
#        state_dict['read_modes'] = tf.nn.softmax(
#            state_dict['read_modes_pre_softmax'],
#            name='read_modes'
#        )
#
#    return state_dict
#
#
#def get_usages(
#    prev_read_weightings,
#    prev_write_weightings,
#    prev_usages,
#    new_dnc_state,
#):
#    # prev_read_weightings: [minibatch_size, R, N]
#    # free_gates: [minibatch_size, R]
#    # TODO: Maybe expand_dims not needed thanks to broadcasting.
#    free_gates = tf.expand_dims(new_dnc_state['free_gates'], -1)
#
#    # memory retention psi_t: [minibatch_size, N]
#    memory_retentions = tf.reduce_prod(
#        (1 - free_gates * prev_read_weightings),
#        axis=[1]
#    )
#
#    # usage u_t: [-1, N]
#    usages = (
#        prev_usages + prev_write_weightings
#        - prev_usages * prev_write_weightings
#    ) * memory_retentions
#
#    return usages
#
#
#def get_allocation_weightings(
#    usages,
#):
#    # free list phi_t: [-1, N]
#    _usages = EPSILON + (1 - EPSILON) * usages
#    non_usages = 1 - _usages
#    sorted_non_usages, sorted_non_usage_indices = tf.nn.top_k(
#        non_usages,
#        k=non_usages.shape.as_list()[-1],
#
#    )
#    sorted_usages = 1 - sorted_non_usages
#    sorted_allocation_weightings = (
#        sorted_non_usages
#        * tf.cumprod(
#            sorted_usages,
#            axis=1,
#            exclusive=True,
#        )
#    )
#    sorted_usage_indices = batch_invert_permutation(sorted_non_usage_indices)
#    allocation_weightings = batch_gather(
#        sorted_allocation_weightings, 
#        sorted_usage_indices,
#    )
#    return allocation_weightings
#
#
#def get_content_weightings(keys, memory, strengths, name=None):
#    # keys: [-1, num_heads, W]
#    # memory: [-1, N, W]
#    # strengths: [-1, num_heads]
#
#    # numerator: [-1, num_heads, N]
#    numerator = tf.matmul(
#        keys,
#        memory,
#    )
#    keys_norm = tf.sqrt(
#        tf.norm(
#            keys,
#            axis=2,
#        )
#    )
#    memory_norm = tf.sqrt(
#        tf.norm(
#            memory,
#            axis=2,
#        )
#    )
#    denominator = tf.einsum(
#        'bh,bn->bhn',
#        keys_norm,
#        memory_norm,
#    ) + EPSILON
#
#    # cosine_similarity: [-1, num_heads, N]
#    cosine_similarity = numerator / denominator
#    
#    content_weighting_logits = tf.einsum(
#        'bhn,bh->bn',
#        cosine_similarity,
#        strengths,
#    )
#    content_weightings = tf.softmax(
#        content_weighting_logits,
#        name=name,
#    )
#    return content_weightings
#
#
#def get_write_weighting(
#    allocation_gates,
#    write_gates,
#    allocation_weightings,
#    write_content_weightings,
#):
#    """
#    write_weightings: [-1, E, N], w_{t}^{w} in the paper.
#    """
#    write_weightings = (
#        write_gates 
#        * (allocation_gates * allocation_weightings
#           + (1 - allocation_gates) * write_content_weightings)
#    )
#    return write_weightings
#

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


# TODO: Rewrite as a subclass of an RNN class
# and use dynamic_rnn instead of using fixed sequence length.
class DifferentiableNeuralComputer:
    def __init__(
        self,
        input_size,
        output_size,
    ):
        self._config = {
            'num_read_heads': 4,   # R, number of read heads.
            'num_write_heads': 1,  # number of write heads = 1 in paper.
            'num_memory_rows': 16,   # N, number of memory rows.
            'width_memory_row': 16,  # W, width of a memory row.
            'num_controller_units': 128,
            'input_size': 128,
            'output_size': 128,
        }
#        self._W_r = # RW \times Y weight matrix

#    def get_zero_state(self):
#        return zero_states

    def _build_graph(self):
        input_size = self._config['input_size']
        output_size = self._config['output_size']
        max_seq_len = self._config['max_sequence_length']
        B = minibatch_size = self._config['minibatch_size']
        N = self._config['num_memory_rows']
        W = self._config['width_memory_row']
        R = self._config['num_read_heads']
        E = self._config['num_write_heads']

        input_seqs = tf.placeholder(
            dtype=tf.float32,
            shape=[B, max_seq_len, input_size]
        )
        inputs = tf.split(
            input_seqs,
            num_or_size_splits=max_seq_len,
            axis=1,
        )
        output_seqs = [None] * max_seq_len

#        # M_t
#        memory = tf.get_variable(
#            name='memory',
#            shape=[B, N, W],
#            initializer=tf.zeros_initializer(dtype=tf.float32),
#        )

        # TODO: Use LSTMBlockCell.
        controller = tf.nn.rnn_cell.LSTMCell(
            num_units=self._config['controller_num_units'],
        )
        controller_prev_states = controller.zero_state(
            batch_size=B,
            dtype=tf.float32,
        )

        prev_read_vectors = 
        # M_t
        prev_memory = tf.zeros(
            shape=[B, N, W],
            dtype=tf.float32,
        )
        for t in range(max_seq_len):
            controller_inputs = tf.concatenate(
                inputs[t],
                prev_read_vectors,
                axis=1,
            )

            # controller_outputs = h_t
            # controller_new_states = s_t
            controller_outputs, controller_new_states = controller(
                controller_inputs,
                controller_prev_states,
            )
            # new interface parameters, \xi_t.
            with tf.variable_scope('interface_parameters') as scope:
                if t > 0:
                    scope.reuse_variables()

                new_interface_dict = build_interface(
                    controller_outputs,
                    num_read_heads=R,
                    num_write_heads=E,
                    width_memory_row=W,
                    num_memory_row=N,
                    variable_initializer=self._get_variable_initializer,
                )

#            # dynamic_memory_allocation, a_t 
#            if t == 0:
#                usages = tf.zeros(
#                    shape=[B, N],
#                    dtype=tf.float32,
#                )
#            else:
#                usages = get_usages(
#                    prev_read_weightings,
#                    prev_write_weightings,
#                    prev_usages,
#                    new_state,
#                )
#            allocation_weightings = get_allocation_weightings(usages)
#
#            # c_{t}^{w}
#            write_content_weightings = get_content_weightings(
#                new_state['write_keys'],
#                memory,
#                new_state['write_strengths'],
#                name='write_content_weightings',
#            )
#
#            # w_{t}^{w}
#            write_weightings = get_write_weightings(
#                new_state['allocation_gates'],
#                new_state['write_gates'],
#                allocation_weightings,
#                write_content_weightings,
#            )
#
            # Update memory with erase & write.
            memory, write_weightings = write_memory(
                prev_read_weightings,
                prev_write_weightings,
                prev_usages,
                prev_memory,
                interface_dict,
                t,
            )

#            # temporal memory link matrix, L_t.
#            if t == 0:
#                precedence_weighting = tf.get_variable(
#                    name='precedence_weighting',
#                    shape=[B, E, N], 
#                    initializer=tf.zeros_initializer(dtype=tf.float32),
#                )
#                temporal_memory_linkage = tf.get_variable(
#                    name='temporal_memory_linkage',
#                    shape=[B, E, N, N], 
#                    initializer=tf.zeros_initializer(dtype=tf.float32),
#                )
#            else:
#                temporal_memory_linkage = get_temporal_memory_linkage(
#                    write_weightings,
#                    prev_precedence_weighting,
#                    prev_temporal_memory_linkage,
#                )
#
#            # f_{t}^{i}: [B, R, E, N]
#            read_forward_weighting = get_directional_read_weightings(
#                temporal_memory_linkage,
#                prev_read_weightings,
#                forward=True,
#            )
#            # b_{t}^{i}: [B, R, E, N]
#            read_backward_weighting = get_directional_read_weightings(
#                temporal_memory_linkage,
#                prev_read_weightings,
#                forward=False,
#            )
#            # c_{t}^{i}: [B, R, E, N]
#            read_content_weightings = get_content_weightings(
#                new_state['read_keys'],
#                memory,
#                new_state['read_strengths'],
#                name='read_content_weightings',
#            )
#            # read weightings w_{t}^{r, i}: [B, R, N].
#            read_weightings = get_read_weightings(
#                new_state['read_modes'],
#                read_backward_weightings,
#                read_content_weightings,
#                read_forward_weightings,
#            )
#
#            # read vector r_{t}^{i}: [B, R, W].
#            read_vectors = tf.matmul(
#                memory,
#                read_weightings,
#            )
            read_vectors = read_memory(
                prev_precedence_weighting,
                prev_temporal_memory_linkage,
                prev_read_weightings,
                write_weightings,
                memory,
                interface_dict,
                t,
            )

            # Preparing the next step.

            outputs[t] = controller_outputs + get_linear_outputs(
                read_vectors,
                output_shape=controller_outputs.shape.as_list(),
                initializer=self._get_variable_initializer(),
                name='dnc_output_{}'.format(t),
            )

            controller_prev_states = controller_new_states

            prev_read_weightings = read_weightings
            prev_write_weightings = write_weightings
            prev_usages = usages
            prev_state = new_state
            prev_precedence_weighting = precedence_weighting
            prev_temporal_memory_linkage = temporal_memory_linkage

        # End of sequence for-loop.

    return output_seqs


#    def run(
#        self,
#        x_t,
#        state,
#    ):
#        prev_rs = self._get_read_vectors()
#        
#        chi_t = concatenate(
#            x_t,
#            prev_rs,
#        )
#        h_t, new_states = self._controller.run(
#            chi_t,
#            prev_states,
#        )
#        hs = concatenate(prev_hs, h_t)
#        v_t = matmul(W_y, hs)
#        xi_t = matmul(W_xi, hs)
#
#        rs = concatenate(prev_rs, r_t)
#        y_t = v_t + matmul(W_r, rs)
#
#        return y_t
#
#
#def read_from_memory(w_r_ts, M_t):
#    r_i_t = matmul(M_t, w_r_i_t)
#    return r_ts
#
#def write_to_memory(M_prev, w_w_t, e_t, nu_t):
#    E = # N \times W matrix of ones.
#    M_t = M_prev * (E - vecmul(w_w_t, e_t)) + vecmul(w_w_t, nu_t)
#
#
#def get_write_weighting(
#    M_prev, k_w_t, beta_w_t,
#    a_t, g_w_t_, g_a_t,
#):
#    C = self.content_lookup
#    c_w_t = C(M_prev, k_w_t, beta_w_t)
#    w_w_t = (
#        g_w_t 
#        * (g_a_t * a_t + (1 - g_a_t) * c_w_t)
#    )
#    return w_w_t
#
#def get_precedence_weighting(p, t, w_ws):
#    # NOTE: p should be initialized with p = {0: 0_vec}.
#    try:
#        return p[t]
#    except KeyError:
#        sum_w_w = 0
#        for i in range(t):
#            sum_w_w += w_ws[i]  
#        p_prev = get_precedence_weighting(p, t -1, w_ws)
#        p[t] = (1 - sum_w_w) * p_prev + w_ws[t]
#        return p[t]
#
## TODO Implement the sparse version.
#def get_temporal_memory_linkage(
#    w_ws, p,
#    L, t,
#):
#    # NOTE: L should be initialized with L = {0: 0_matrix}.
#    try:
#        return L[t]:
#    except KeyError:
#        w_t = w_ws[t]
#        L_prev = get_temporal_memory_linkage(w_ws, p, L, t - 1)
#        p_prev = get_precedence_weighting(p, t - 1, w_ws)
#        L_t = zeros(shape=(N, N))
#        for i in range(N):
#            for j in range(N):
#                L_t[i][j] = (
#                    (1 - w_w_t[i] = w_w_t[j]) * L_prev
#                    + w_w_t[i] * p_prev[j]
#                )
#        L[t] = L_t
#        return L[t]
#
#def get_read_weighting(
#    L, w_ws, p, 
#    w_r_prevs,
#    M_t, pi_ts,
#    t,
#)
#    L_t = get_temporal_memory_linkage(w_ws, p, L, t)
#    L_t_T = transpose(L_t)
#    for i in range(R):
#        w_r_prev_i = w_r_prevs[i]
#        f_t[i] = matmul(L_t, w_r_prev_i)
#        b_t[i] = matmul(L_t_T, w_r_prev_i)
#
#    C = self.content_lookup
#    for i in range(R):
#        c_r_t[i] = C(M_t, k_r_t[i], beta_r_t[i])
#        w_r_t[i] = (
#            pi_r[i][1] * b_t[i]
#            + pi_r[i][2] * c_t[i]
#            + pi_r[i][3] * f_t[i]
#        )
#    return w_r_t


