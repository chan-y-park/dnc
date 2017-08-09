import numpy as np
import tensorflow as tf

DNC_NUM_READ_MODES = 3
EPSILON = 1e-6

class DNCLSTMController:
    def __init__(self):
        pass

    def run(
        chi_t,
        prev_states
    ):
        return h_t, new_states


def get_linear_outputs(
    inputs,
    outputs_shape,
    initializer,
    name,
):
    # inputs: [B, controller_output_size]
    inputs_size = inputs.shape.as_list()[-1]
    linear_size = np.prod(outputs_shape)
    with tf.variable_scope(name):
        W = tf.get_variable(
            name='kernel',
            shape=[inputs_size, linear_size],
            initializer=variable_initializer(),
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
            shape=([-1] + outputs_shape),
            name=name,
        )
    return outputs


def build_dnc_state(
    controller_outputs,
#    minibatch_size,
    num_read_heads,
    num_write_heads,
    width_memory_row,
    num_memory_row,
    variable_initializer,
):
#    B = minibatch_size
    N = num_memory_row
    W = width_memory_row
    R = num_read_heads
    E = num_write_heads

    state_dict = {}
    with tf.variable_scope('interface_parameters'):
        # k^{r, i}_t
        state_dict['read_keys'] = get_linear_outpus(
            controller_outputs,
            outputs_shape=[R, W],
            initializer=variable_initializer(),
            name='read_keys',
        )
        # \hat{beta}^{r, i}_t
        state_dict['read_strengths_pre_oneplus'] = get_linear_outpus(
            controller_outputs,
            shape=[R],
            initializer=variable_initializer(),
            name='read_strength_pre_oneplus',
        )
        state_dict['read_strengths'] = oneplus(
            state_dict['read_strengths_pre_oneplus']
            name='read_strengths',
        )
        # k^{w}_t
        state_dict['write_key'] = get_linear_outpus(
            controller_outputs,
            outputs_shape=[W],
            initializer=variable_initializer(),
            name='write_key',
        )
        # \hat{beta}^{w}_t
        state_dict['write_strength_pre_oneplus'] = get_linear_outpus(
            controller_outputs,
            outputs_shape=[E],
            initializer=variable_initializer(),
            name='write_strength_pre_oneplus',
        )
        state_dict['write_strength'] = oneplus(
            state_dict['write_strength_pre_oneplus']
            name='write_strength',
        )
        # \hat{e}_t
        state_dict['erase_pre_sigmoid'] = get_linear_outpus(
            controller_outputs,
            outputs_shape=[W],
            initializer=variable_initializer(),
            name='erase_pre_sigmoid',
        )
        state_dict['erase'] = tf.sigmoid(
            state_dict['erase_pre_sigmoid'],
            name='erase',
        )
        # \nu_t
        state_dict['write'] = get_linear_outpus(
            controller_outputs,
            outputs_shape=[W],
            initializer=self._get_variable_initializer(),
            name='write',
        )
        # f^i_t
        state_dict['free_gates_pre_sigmoid'] = get_linear_outpus(
            controller_outputs,
            outputs_shape=[R],
            initializer=self._get_variable_initializer(),
            name='free_gates_pre_sigmoid',
        )
        state_dict['free_gates'] = tf.sigmoid(
            self._free_gates_pre_sigmoid,
            name='free_gates',
        )
        # g^a_t
        state_dict['allocation_gates_pre_sigmoid'] = get_linear_outpus(
            controller_outputs,
            outputs_shape=[E],
            initializer=variable_initializer(),
            name='allocation_gates_pre_sigmoid',
        )
        state_dict['allocation_gates'] = tf.sigmoid(
            state_dict['allocation_gates_pre_sigmoid'],
            name='allocation_gates',
        )
        # g^w_t
        state_dict['write_gates_pre_sigmoid'] = get_linear_outpus(
            controller_outputs,
            outputs_shape=[E],
            initializer=get_variable_initializer(),
            name='write_gates_pre_sigmoid',
        )
        state_dict['write_gates'] = tf.sigmoid(
            state_dict['write_gates_pre_sigmoid'],
            name='write_gates',
        )
        # pi^i_t
        state_dict['read_modes_pre_softmax'] = get_linear_outpus(
            controller_outputs,
            outputs_shape=[R, DNC_NUM_READ_MODES],
            initializer=self._get_variable_initializer(),
            name='read_modes_pre_softmax',
        )
        state_dict['read_modes'] = tf.nn.softmax(
            self._read_modes_pre_softmax,
            name='read_modes'
        )

    return state_dict


def build_dynamic_memory_allocation(
    prev_read_weightings,
    prev_usages,
    new_dnc_state,
):
    # prev_read_weightings: [minibatch_size, R, N]
    # free_gates: [minibatch_size, R]
    # TODO: Maybe expand_dims not needed thanks to broadcasting.
    free_gates = tf.expand_dims(new_dnc_state['free_gates'], -1)

    # memory retention psi_t: [minibatch_size, N]
    memory_retentions = tf.reduce_prod(
        (1 - free_gate * prev_read_weightings),
        axis=[1]
    )

    # usage u_t: [-1, N]
    usages = (
        prev_usages + prev_write_weighting
        - prev_usages * prev_write_weightings
    ) * memory_retentions

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
            sorted_usage,
            axis=1,
            exclusive=True,
        )
    )
    sorted_usage_indices = batch_invert_permutation(sorted_non_usage_indices)
    allocation_weightings = batch_gather(
        sorted_allocation_weightingss, 
        sorted_usage_indices,
    )
    return allocation_weightings


def get_content_weightings(keys, memory, strengths, name=None):
    # keys: [-1, num_heads, W]
    # memory: [-1, N, W]
    # strengths: [-1, num_heads]

    # numerator: [-1, num_heads, N]
    numerator = tf.matmul(
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
    denominator = tf.einsum(
        'bh,bn->bhn',
        keys_norm,
        memory_norm,
    ) + EPSILON

    # cosine_similarity: [-1, num_heads, N]
    cosine_similarity = numerator / denominator
    
    content_weighting_logits = tf.einsum(
        'bhn,bh->bn',
        cosine_similarity,
        strengths,
    )
    content_weightings = tf.softmax(
        content_weighting_logits,
        name=name,
    )
    return content_weightings


def get_write_weighting(
    allocation_gates,
    write_gates,
    allocation_weightings,
    write_content_weightings,
)
    write_weightings = (
        write_gates 
        * (allocation_gates * allocation_weightings
           + (1 - allocation_gates) * write_content_weightgins)
    )
    return write_weightings

# TODO: Rewrite as a subclass of an RNN class.
class DifferentiableNeuralComputer:
    def __init__(
        self,
        controller='lstm',
        input_size,
        output_size,
    ):
        self._config = {
            'num_read_vectors': 4,   # R, number of read vectors.
#            'num_write_vectors': 1,  # number of write vectors = 1 in paper.
            'num_memory_rows': 16,   # N, number of memory rows.
            'width_memory_row': 16,  # W, width of a memory row.
            'num_controller_units': ,
            'input_size': ,
            'output_size': ,
        }
        self._W_r = # RW \times Y weight matrix

    def get_zero_state(self):
        return zero_states

    def _build_graph(self):
        minibatch_size = self._config['minibatch_size']
        input_size = self._config['input_size']
        output_size = self._config['output_size']
        N = self._config['num_memory_rows']
        W = self._config['width_memory_row']
        R = self._config['num_read_vectors']
        inputs = tf.placeholder(
            dtype=tf.float32,
            shape=[minibatch_size, input_size]
        )
        # M_t
        self._memory = tf.get_variable(
            name='memory',
            shape=[N, W],
            initializer=tf.zeros_initializer(dtype=tf.float32)
        )
        with tf.variable_scope('controller') as scope:
            # TODO: Use LSTMBlockCell.
            self._controller = tf.nn.rnn_cell.LSTMCell(
                num_units=self._config['controller_num_units'],
            )
            controller_zero_state = self._controller(
                batch_size=,
                dtype=tf.float32,
            )
            controller_outputs, controller_new_states = self._controller(
                inputs,
                controller_zero_states,
            )

            scope.reuse_variables()

        with tf.variable_scope('new_state'):
            self._new_state = build_dnc_state(
                num_read_heads=R,
                num_write_heads=1,
                width_memory_row=W,
                num_memory_row=N,
                variable_initializer=self._get_variable_initializer,
            )

        with tf.variable_scope('dynamic_memory_allocation')
            allocation_weightings = (
                build_dynamic_memory_allocation(
                    self._prev_state['read_weightings'],
                    self._prev_state['usages'],
                    self._new_state,
                )
            )

        write_content_weightings = get_content_weightings(
            tf.expand_dims(self._new_state['write_keys'],
            self._memory,
            self._new_state['write_strengths'],
            name='write_content_weightings',
        )

        write_weighting = get_write_weightings(
            self._new_state['allocation_gates'],
            self._new_state['write_gates'],
            allocation_weightings,
            write_content_weightings,
        )

        with tf.variable_scope('write_weighting')

        with tf.variable_scope('temporal_memory_linkage')

        with tf.variable_scope('read_weighting')



    def run(
        self,
        x_t,
        state,
    ):
        prev_rs = self._get_read_vectors()
        
        chi_t = concatenate(
            x_t,
            prev_rs,
        )
        h_t, new_states = self._controller.run(
            chi_t,
            prev_states,
        )
        hs = concatenate(prev_hs, h_t)
        v_t = matmul(W_y, hs)
        xi_t = matmul(W_xi, hs)

        rs = concatenate(prev_rs, r_t)
        y_t = v_t + matmul(W_r, rs)

        return y_t


def read_from_memory(w_r_ts, M_t):
    r_i_t = matmul(M_t, w_r_i_t)
    return r_ts

def write_to_memory(M_prev, w_w_t, e_t, nu_t):
    E = # N \times W matrix of ones.
    M_t = M_prev * (E - vecmul(w_w_t, e_t)) + vecmul(w_w_t, nu_t)


def get_write_weighting(
    M_prev, k_w_t, beta_w_t,
    a_t, g_w_t_, g_a_t,
):
    C = self.content_lookup
    c_w_t = C(M_prev, k_w_t, beta_w_t)
    w_w_t = (
        g_w_t 
        * (g_a_t * a_t + (1 - g_a_t) * c_w_t)
    )
    return w_w_t

def get_precedence_weighting(p, t, w_ws):
    # NOTE: p should be initialized with p = {0: 0_vec}.
    try:
        return p[t]
    except KeyError:
        sum_w_w = 0
        for i in range(t):
            sum_w_w += w_ws[i]  
        p_prev = get_precedence_weighting(p, t -1, w_ws)
        p[t] = (1 - sum_w_w) * p_prev + w_ws[t]
        return p[t]

# TODO Implement the sparse version.
def get_temporal_memory_linkage(
    w_ws, p,
    L, t,
):
    # NOTE: L should be initialized with L = {0: 0_matrix}.
    try:
        return L[t]:
    except KeyError:
        w_t = w_ws[t]
        L_prev = get_temporal_memory_linkage(w_ws, p, L, t - 1)
        p_prev = get_precedence_weighting(p, t - 1, w_ws)
        L_t = zeros(shape=(N, N))
        for i in range(N):
            for j in range(N):
                L_t[i][j] = (
                    (1 - w_w_t[i] = w_w_t[j]) * L_prev
                    + w_w_t[i] * p_prev[j]
                )
        L[t] = L_t
        return L[t]

def get_read_weighting(
    L, w_ws, p, 
    w_r_prevs,
    M_t, pi_ts,
    t,
)
    L_t = get_temporal_memory_linkage(w_ws, p, L, t)
    L_t_T = transpose(L_t)
    for i in range(R):
        w_r_prev_i = w_r_prevs[i]
        f_t[i] = matmul(L_t, w_r_prev_i)
        b_t[i] = matmul(L_t_T, w_r_prev_i)

    C = self.content_lookup
    for i in range(R):
        c_r_t[i] = C(M_t, k_r_t[i], beta_r_t[i])
        w_r_t[i] = (
            pi_r[i][1] * b_t[i]
            + pi_r[i][2] * c_t[i]
            + pi_r[i][3] * f_t[i]
        )
    return w_r_t


def oneplus(tensor, name):
    return tf.add(
        tf.softplus(tensor),
        1.0,
        name=name,
    )

# NOTE: From deepmind/dnc/utils.py
def batch_invert_permutation(permutations):
  """Returns batched `tf.invert_permutation` for every row in `permutations`."""
  with tf.name_scope('batch_invert_permutation', values=[permutations]):
    unpacked = tf.unstack(permutations)
    inverses = [tf.invert_permutation(permutation) for permutation in unpacked]
    return tf.stack(inverses)


# NOTE: From deepmind/dnc/utils.py
def batch_gather(values, indices):
  """Returns batched `tf.gather` for every row in the input."""
  with tf.name_scope('batch_gather', values=[values, indices]):
    unpacked = zip(tf.unstack(values), tf.unstack(indices))
    result = [tf.gather(value, index) for value, index in unpacked]
    return tf.stack(result)
