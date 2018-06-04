import tensorflow as tf


def conv2d(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           padding='SAME',
           activation=tf.nn.relu,
           stddev=0.01,
           bias=0.0,
           trainable=True,
           scope=None,
           reuse=None):
    """Adds a 2D convolution followed by an optional batch_norm layer.
  conv2d creates a variable called 'weights', representing the convolutional
  kernel, that is convolved with the input. If `batch_norm_params` is None, a
  second variable called 'biases' is added to the result of the convolution
  operation.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_filters_out: the number of output filters.
    kernel_size: a list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    stride: a list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: one of 'VALID' or 'SAME'.
    activation: activation function.
    stddev: standard deviation of the truncated guassian weight distribution.
    bias: the initial value of the biases.
    trainable: whether or not the variables should be trainable or not.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
  Returns:
    a tensor representing the output of the operation.
  """
    with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse):

        kernel_h, kernel_w = kernel_size[0], kernel_size[1]
        stride_h, stride_w = (stride, stride)

        num_filters_in = inputs.get_shape()[-1]
        weights_shape = [kernel_h, kernel_w,
                         num_filters_in, num_filters_out]
        weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
        weights = tf.get_variable("weights", shape=weights_shape, dtype=tf.float32, initializer=weights_initializer,
                                  trainable=trainable)

        conv = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1],
                            padding=padding)

        bias_shape = [num_filters_out, ]
        bias_initializer = tf.constant_initializer(bias)
        biases = tf.get_variable("biases", shape=bias_shape, dtype=tf.float32, initializer=bias_initializer,
                                 trainable=trainable)
        outputs = tf.nn.bias_add(conv, biases)
        if activation:
            outputs = activation(outputs)
        return outputs


def fc(inputs,
       num_units_out,
       activation=tf.nn.relu,
       stddev=0.01,
       bias=0.0,
       trainable=True,
       scope=None):
    """
    fully connected layers
    :param inputs: input tensor
    :param num_units_out: number of output units
    :param activation: activation function
    :param stddev: standard deviation for the weights
    :param bias: the initial value of biases
    :param trainable: whether variable should be trained
    :param scope:
    :return: the tensor variable
    """
    with tf.variable_scope(scope, 'FC', [inputs]):
        num_units_in = inputs.get_shape()[1]
        weights_shape = [num_units_in, num_units_out]
        weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
        weights = tf.get_variable("weights", shape=weights_shape, dtype=tf.float32, initializer=weights_initializer,
                                  trainable=trainable)
        biases_shape = [num_units_out, ]
        biases_initializer = tf.constant_initializer(bias)
        biases = tf.get_variable("biases", shape=biases_shape, dtype=tf.float32, initializer=biases_initializer,
                                 trainable=trainable)
        outputs = tf.nn.xw_plus_b(inputs, weights, biases)
        if activation:
            outputs = activation(outputs)
        return outputs
