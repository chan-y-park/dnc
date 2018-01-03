#import numpy as np
import tensorflow as tf

from dnc_controller_interface import build_interface
from dnc_memory_write import write_memory
from dnc_memory_read import read_memory
from utils import get_linear_outputs

default_config = {
    'num_read_heads': 4,        # R, number of read heads.
    'num_write_heads': 1,       # number of write heads = 1 in paper.
    'num_memory_rows': 16,      # N, number of memory rows.
    'width_memory_row': 10,     # W, width of a memory row.
    'controller_num_units': 32,
    'input_size': 8,
    'output_size': 8,
    'max_sequence_length': 5,
    'minibatch_size': 2,
    'variable_initializer': {
        'mean': 0,
        'stddev': 0.02,
    },
    'epsilon': 1e-6,
}


# TODO: Rewrite as a subclass of an RNN class
# and use dynamic_rnn instead of using fixed sequence length.
class DifferentiableNeuralComputer:
    def __init__(
        self,
    ):
        self._config = default_config

    def _get_variable_initializer(self):
        return tf.truncated_normal_initializer(
            dtype=tf.float32,
            **self._config['variable_initializer']
        )

    def _build_graph(self, inputs):
        B = minibatch_size = self._config['minibatch_size']
        N = self._config['num_memory_rows']
        W = self._config['width_memory_row']
        R = self._config['num_read_heads']
        E = self._config['num_write_heads']

        # TODO: Use LSTMBlockCell.
        with tf.variable_scope('controller'):
            controller = tf.nn.rnn_cell.LSTMCell(
                num_units=self._config['controller_num_units'],
            )
            controller_prev_states = controller.zero_state(
                batch_size=B,
                dtype=tf.float32,
            )

        prev_usages = None 
        prev_write_weightings = None 
        prev_memory = tf.zeros(
            shape=[B, N, W],
            dtype=tf.float32,
        )

        prev_precedence_weightings = None 
        prev_temporal_memory_linkage = None 
        prev_read_weightings = None

        prev_read_vectors = tf.zeros(
            shape=[B, R, W],
            dtype=tf.float32,
        )

        flattened_prev_read_vectors = tf.reshape(
            prev_read_vectors,
            shape=[B, R * W],
        )
        controller_inputs = tf.concat(
            [inputs, flattened_prev_read_vectors],
            axis=1,
        )
        
        with tf.variable_scope('controller', reuse=scope_reuse):
            # controller_outputs = h_t
            # controller_new_states = s_t
            controller_outputs, controller_new_states = controller(
                controller_inputs,
                controller_prev_states,
            )
        # new interface parameters, \xi_t.
        with tf.variable_scope('interface_parameters', reuse=scope_reuse):
            interface_dict = build_interface(
                controller_outputs,
                minibatch_size=B,
                num_read_heads=R,
                num_write_heads=E,
                width_memory_row=W,
                num_memory_row=N,
                variable_initializer=self._get_variable_initializer(),
            )

        # Update memory with erase & write.
        usages, write_weightings, memory = write_memory(
            prev_read_weightings,
            prev_write_weightings,
            prev_usages,
            prev_memory,
            interface_dict,
            self._config,
            t,
        )

        (precedence_weightings, temporal_memory_linkage, 
         read_weightings, read_vectors) = read_memory(
            prev_precedence_weightings,
            prev_temporal_memory_linkage,
            prev_read_weightings,
            write_weightings,
            memory,
            interface_dict,
            self._config,
            t,
        )

        # Preparing the next step.

        # NOTE: outputs shape = [B, output_size]
        outputs = controller_outputs + get_linear_outputs(
            tf.reshape(read_vectors, shape=[B, R * W]),
            outputs_shape=controller_outputs.shape.as_list(),
            variable_initializer=self._get_variable_initializer(),
            name='dnc_output_{}'.format(t),
        )

        controller_prev_states = controller_new_states

        prev_usages = usages
        prev_write_weightings = write_weightings
        prev_memory = memory

        prev_precedence_weightings = precedence_weightings
        prev_temporal_memory_linkage = temporal_memory_linkage
        prev_read_weightings = read_weightings

        return outputs
