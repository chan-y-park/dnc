import tensorflow as tf

X = # input vector size
Y = # output vector size
R = # number of read vectors
N = # number of memory rows
W = # width of a memory row

class DNCLSTMController:
    def __init__(self):

    def run(
        chi_t,
        prev_states
    ):
        return h_t, new_states

class InterfaceParameter:
    def __init__(self):
        self.k_r_ts         # R read keys
        self.hat_beta_r_ts  # R read strengths
        self.k_w_t,         # the write key
        self.hat_beta_w_t,  # the write strength pre-oneplus
        self.hat_e_t        # the erase vector pre-sigmoid
        self.nu_t            # the write vector
        self.hat_f_ts       # R free gates pre-sigmoid
        self.hat_g_a_t      # the allocation gate pre-sigmoid
        self.hat_g_w_t      # the write gate pre-sigmoid
        self.hat_pi_ts      # R read modes pre-softmax

class DNCWriteHead:

class DNCReadHead:

class DNCMemory:

class DifferentiableNeuralComputer:
    def __init__(
        self,
        controller='lstm',
    ):
        self._W_r = # RW \times Y weight matrix
        # Prepare a controller.
        if controller == 'lstm':
            self._controller = DNCLSTMController()
        # Prepare memory.

    def get_zero_state(self):
        return zero_states

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

def content_lookup_op(M, k, beta):
    D = self.cosine_similarity
    denominator = 0
    for M_j in M:
        denominator += exp(D(k, M_j, beta)
    C = zeros(shape=(N))
    for i in range(N):
        C[i] = (
            exp(D(k, M[i], beta))
            / denominator
        )
    return C

    
def cosine_similarity(u, v):
    return (
        vecmul(u, v)
        / (norm(u) * norm(v))
    )

def dynamic_memory_allocation(
    f_ts, w_r_prevs,
    u_prev, w_w_prev,
):
    psi_t = 1
    for i in enumerate(f_ts):
        f_i = f_ts[i]
        w_r_prev_i = w_r_prevs[i]
        psi_t *= (1 - vecmul(f_i, w_r_prev_i)

    # u_t \in [0, 1]^N, u_0 = 0
    u_t = (u_prev + w_w_prev - u_prev * w_w_prev) * psi_t

    phi_t = argsort(u_t)
    a_t = zeros(shape=(N))
    for j in range(N):
        prods = 1
        for i in range(j):
            prods *= u_t[phi_t[i]]
        a_t[phi_t[j]] = (
            (1 - u_t[phi_t[j]])
            * prods
        )

    return a_t

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
