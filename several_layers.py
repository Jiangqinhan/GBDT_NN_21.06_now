import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K


class mmoe_layer(tf.keras.layers.Layer):
    """
    The Multi-gate Mixture-of-Experts layer in MMOE model
      Input shape
        - 2D tensor with shape: ``(batch_size,units)``.
      Output shape
        - A list with **num_tasks** elements, which is a 2D tensor with shape: ``(batch_size, output_dim)`` .
      Arguments
        - **num_tasks**: integer, the number of tasks, equal to the number of outputs.
        - **num_experts**: integer, the number of experts.
        - **output_dim**: integer, the dimension of each output of MMOELayer.
    References
      - [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
    """

    def __init__(self, num_tasks, num_experts, output_dim, seed=1024, **kwargs):
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.seed = seed
        super(mmoe_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        emb_size = int(input_shape[-1])
        self.experts_kernel = self.add_weight(name='experts_kernel',
                                              shape=(emb_size, self.num_experts * self.output_dim), dtype=tf.float32,
                                              initializer=tf.initializers.glorot_normal(seed=self.seed))
        self.gate_kernels = []
        for i in range(self.num_tasks):
            self.gate_kernels.append(
                self.add_weight(name='gate_kernel_{}'.format(i), shape=(emb_size, self.num_experts),
                                initializer=tf.initializers.glorot_normal(seed=self.seed), dtype=tf.float32))
        super(mmoe_layer, self).build(input_shape)

    def call(self, input, **kwargs):
        experts_out = tf.tensordot(input, self.experts_kernel, axes=(-1, 0))
        gate_scores = [tf.tensordot(input, x, axes=(-1, 0)) for x in self.gate_kernels]
        experts_out = tf.reshape(experts_out, [-1, self.num_experts, self.output_dim])
        output = []
        for i in range(self.num_tasks):
            gate_score = tf.tensordot(input, self.gate_kernels[i], axes=(-1, 0))
            gate_score = tf.nn.softmax(gate_score)
            gate_score = tf.expand_dims(gate_score, axis=2)
            gate_score = tf.tile(gate_score, [1, 1, self.output_dim])
            tmp = tf.multiply(gate_score, experts_out)
            output.append(tf.reduce_sum(tmp, axis=1))
        return output

    def get_config(self):

        config = {'num_tasks': self.num_tasks,
                  'num_experts': self.num_experts,
                  'output_dim': self.output_dim}
        base_config = super(mmoe_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_dim] * self.num_tasks


class CrossNetMix(tf.keras.layers.Layer):
    """The Cross Network part of DCN-Mix model:

      Input shape
        - 2D tensor with shape: ``(batch_size, units)``

      Output shape
        - 2D tensor with shape: ``(batch_size, units)``

      Arguments
        - **low_rank** : Positive integer, dimensionality of low-rank sapce

        - **num_experts** : Positive integer, number of experts.

        - **layer_num**: Positive integer, the cross layer number

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix

        - **seed**: A Python integer to use as random seed.

    """

    def __init__(self, low_rank=32, num_experts=4, layer_num=2, l2_reg=0, seed=1024, **kwargs):
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.layer_num = layer_num
        self.seed = seed
        self.l2_reg = l2_reg
        super(CrossNetMix, self).__init__(**kwargs)

    def build(self, input_shape):
        emb_size = int(input_shape[-1])
        self.U_list = [
            self.add_weight(name="U_{}".format(i), shape=(self.num_experts, emb_size, self.low_rank), dtype=tf.float32,
                            initializer=tf.initializers.truncated_normal(self.seed),
                            regularizer=tf.keras.regularizers.l2(self.l2_reg)) for i in range(self.layer_num)]
        # all expert share the same C
        self.C_list = [
            self.add_weight(name="C_{}".format(i), shape=(self.low_rank, self.low_rank), dtype=tf.float32,
                            initializer=tf.initializers.truncated_normal(self.seed),
                            regularizer=tf.keras.regularizers.l2(self.l2_reg)) for i in range(self.layer_num)]
        self.V_list = [
            self.add_weight(name="U_{}".format(i), shape=(self.num_experts, self.low_rank, emb_size), dtype=tf.float32,
                            initializer=tf.initializers.truncated_normal(self.seed),
                            regularizer=tf.keras.regularizers.l2(self.l2_reg)) for i in range(self.layer_num)]
        self.gate_list = [
            self.add_weight(name="gate_{}".format(i), shape=(emb_size, self.num_experts),
                            initializer=tf.initializers.truncated_normal(self.seed), dtype=tf.float32)
            for i in range(self.layer_num)
        ]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(1,emb_size),
                                     initializer=tf.initializers.zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        super(CrossNetMix, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))
        # tf.einsum
        # x_0=tf.expand_dims(inputs,axis=2)
        x0 = inputs
        xl = inputs
        for i in range(self.layer_num):
            gate_score = tf.tensordot(inputs, self.gate_list[i], axes=(-1, 0))
            gate_score = tf.nn.softmax(gate_score)
            expert_out = []
            for expertid in range(self.num_experts):
                U_X = tf.tensordot(xl, self.U_list[i][expertid], axes=(-1, 0))
                U_X = tf.nn.tanh(U_X)
                U_X = tf.tensordot(U_X, self.C_list[i], axes=(-1, 0))
                U_X = tf.nn.tanh(U_X)
                V_U = tf.tensordot(U_X, self.V_list[i][expertid], axes=(-1, 0))
                V_U += self.bias[i]
                tmp = tf.multiply(V_U, x0)
                expert_out.append(V_U)
            expert_out = tf.stack(expert_out, axis=2)  # (batch_size,emb_size)->(batch_size,emb_size,expert_num)
            gate_score = tf.expand_dims(gate_score, axis=2)  # (batch_size,expert_num)->(batch_size,expert_num,1)
            mmo_out = tf.matmul(expert_out, gate_score)
            mmo_out = tf.squeeze(mmo_out, axis=2)
            xl += mmo_out
        return xl
