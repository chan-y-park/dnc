import numpy
import tensorflow as tf


class CopyTask:
    def __init__(
        self,
        config,
    ):
        self._config = config,
        self._tf_config = tf.ConfigProto()
        self._tf_config.gpu_options.allow_growth = (
            self._config['gpu_options']['allow_memory_growth']
        )
        self._tf_graph = tf.Graph()
        with tf_graph.as_default():
            self._build_graph()

            self._tf_session = tf.Session(config=self._tf_config)
            self._tf_session.run(
                tf.variable_initializer(
                    self._tf_graph.get_collection('variables')
                )
            )
            self._tf_saver = tf.train.Saver(
                var_list=self._tf_graph.get_collection('trainable_variables'),
            )

    def restore(self, checkpoint_save_path):
        self._tf_saver.restore(
            self._tf_session,
            checkpoint_save_path,
        )

    def _build_graph(self):
        a_dnc = DifferentiableNeuralComputer()

        B = minibatch_size = self._config['minibatch_size']
        input_size = self._config['input_size']
        output_size = self._config['output_size']
        max_seq_len = self._config['max_sequence_length']

        input_seqs = tf.placeholder(
            dtype=tf.float32,
            shape=[B, max_seq_len, input_size]
            name='input_seqs',
        )
        outputs = [None] * max_seq_len

        for t in range(max_seq_len):
            if t == 0:
                scope_reuse = None 
            else:
                scope_reuse = True

            # NOTE: inputs shape = [B, input_size]
            inputs = input_seqs[:, t, :]
            with tf.variable_scope('dnc_{}'.format(t)):
                outputs[t] = a_dnc._build_graph(inputs)

        # NOTE: output_seqs shape = [B, max_seq_len, output_size]
        output_seqs = tf.stack(
            outputs,
            axis=1,
            name='output_seqs',
        )
        target_seqs = tf.placeholder(
            dtype=tf.float32,
            shape=[B, max_seq_len, output_size]
            name='target_seqs',
        )
        with tf.variable_scope('train'):
            num_steps = tf.get_variable(
                name='step',
                shape=[],
                dtype=tf.int32,
            )
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=output_seqs,
                labels=target_seqs,
                name='losses',
            )
            minibatch_loss = tf.reduce_sum(
                losses,
                name='minibatch_loss',    
            )
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=1e-4,
            )
            train_var_list = []
            for var in tf_graph.get_collection('trainable_variables'):
                train_var_list.append(var)
            grads_and_vars = optimizer.compute_gradients(
                var_list=train_var_list,
                loss=minibatch_loss,
            )
            # TODO: gradient clipping, see img2txt.
            train_op = optimizer.apply_gradients(
               grads_and_vars,
               name='minimize_loss',
            )

    def test_train(
        self,
        max_num_steps,    
    ):
        pattern_size = self._config['pattern_size']
        B = minibatch_size = self._config['minibatch_size']
        input_size = self._config['input_size']
        output_size = self._config['output_size']
        max_seq_len = self._config['max_sequence_length']

        input_seqs = np.zeros(
            shape=(B, max_seq_len, input_size),
            dtype=np.float32,
        )
        assert(output_size == input_size)
        target_seqs = np.zeros(
            shape=(B, max_seq_len, output_size),
            dtype=np.float32,
        )

        for i_pattern in range(max_seq_len // (2 * pattern_size)):
            a_pattern = np.random.randint(
                low=0,
                high=2,
                size=(B, pattern_size, input_size)    
            )
            offset = i_pattern
            input_seq[:, offset:(offset + pattern_size), :] = a_pattern
            offset = i_pattern + pattern_size
            target_seq[:, offset:(offset + pattern_size), :] = a_pattern

        feed_dict = {
            self._tf_graph.get_tensor_by_name('input_seqs:0'): input_seqs,
            self._tf_graph.get_tensor_by_name('target_seqs:0'): target_seqs,
        }

        fetch_dict = {}
        for var_name in [
            'output_seqs',
            'losses',
            'minibatch_loss',
        ]:
            fetch_dict[var_name] = self._tf_graph.get_tensor_by_name(
                var_name + ':0'
            )
        for op_name in [
            'minimize_loss',
        ]:
            fetch_dict[op_name] = self._tf_graph.get_operation_by_name(op_name)
            
        step = 0
        while step < max_num_steps:
            step_op = tf.assign(
                self._tf_graph.get_variable_by_name('train/step:0'),
                step,
            )

            self._tf_session.run(
                feed_dict=feed_dict,
                fetches=fetch_dict,
            )

        return {
            'feed_dict': feed_dict,
            'fetch_dict': fetch_dict,
        }

