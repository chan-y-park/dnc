#import numpy as np
import tensorflow as tf

from dnc_controller_interface import build_interface
from dnc_memory_write import write_memory
from dnc_memory_read import read_memory
from utils import get_linear_outputs

EPSILON = 1e-6


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
        outputs = [None] * max_seq_len

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

        prev_usages = None 
        prev_write_weightings = None 
        prev_memory = tf.zeros(
            shape=[B, N, W],
            dtype=tf.float32,
        )

        prev_precedence_weighting = None 
        prev_temporal_memory_linkage = None 
        prev_read_weightings = None

        prev_read_vectors = tf.zeros(
            shape=[B, R, W],
            dtype=tf.float32,
        )
        for t in range(max_seq_len):
            # NOTE: inputs shape = [B, input_size]
            inputs = input_seqs[:, t, :]
            controller_inputs = tf.concatenate(
                inputs,
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

                interface_dict = build_interface(
                    controller_outputs,
                    num_read_heads=R,
                    num_write_heads=E,
                    width_memory_row=W,
                    num_memory_row=N,
                    variable_initializer=self._get_variable_initializer,
                )

            # Update memory with erase & write.
            usages, write_weightings, memory = write_memory(
                prev_read_weightings,
                prev_write_weightings,
                prev_usages,
                prev_memory,
                interface_dict,
                t,
            )

            (precedence_weighting, temporal_memory_linkage, 
             read_weightings, read_vectors) = read_memory(
                prev_precedence_weighting,
                prev_temporal_memory_linkage,
                prev_read_weightings,
                write_weightings,
                memory,
                interface_dict,
                t,
            )

            # Preparing the next step.

            # NOTE: outputs shape = [B, output_size]
            outputs[t] = controller_outputs + get_linear_outputs(
                read_vectors,
                output_shape=controller_outputs.shape.as_list(),
                initializer=self._get_variable_initializer(),
                name='dnc_output_{}'.format(t),
            )

            controller_prev_states = controller_new_states

            prev_usages = usages
            prev_write_weightings = write_weightings
            prev_memory = memory

            prev_precedence_weighting = precedence_weighting
            prev_temporal_memory_linkage = temporal_memory_linkage
            prev_read_weightings = read_weightings

        # End of sequence for-loop.

        # NOTE: output_seqs shape = [B, max_seq_len, output_size]
        output_seqs = tf.stack(
            outputs,
            axis=1,
        )

        return output_seqs
