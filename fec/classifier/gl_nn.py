import graphlab as gl


class GraphLabNeuralNetBuilder(object):
    """Wrapper for GraphLab NeuralNet class to ease construction of NN
    """
    def __init__(self):
        self.layers = list()
        self.net = gl.deeplearning.NeuralNet()

    def get_net(self):
        """Return the net

        This method verifies that the internal net is valid. If it isn't
        an exception is thrown.

        :return: graphlab nn
        """
        self.net.layers = self.layers
        self.net.verify()
        return self.net

    def verify(self):
        """Verify that the underlying neural net is valid

        :return: True otherwise an exception is thrown
        """
        self.net.layers = self.layers
        return self.net.verify()

    def add_convolution_layer(self, kernel_size,
                              stride, num_channels, **kwargs):
        """Add a convolution layer to the net

        :param kernel_size: size of convolution kernel
        :param stride: stride of kernel
        :param num_channels: number of output filters
        :param kwargs:
        """
        conv_layer = gl.deeplearning.layers.ConvolutionLayer(
            kernel_size=kernel_size, num_channels=num_channels,
            stride=stride, **kwargs
        )
        self.layers.append(conv_layer)

    def add_max_pooling_layer(self, kernel_size, stride=1, padding=0):
        """Add a max pooling layer to the neural net

        :param kernel_size: size of the max pooling layer
        :param stride: length of stride between kernels
        :param padding: number of padding pixels around the data
        """
        pool_layer = gl.deeplearning.layers.MaxPoolingLayer(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.layers.append(pool_layer)

    def add_avg_pooling_layer(self, kernel_size, stride=1, padding=0):
        """Add an average pooling layer to the neural net

        :param kernel_size: size of the average pooling layer
        :param stride: length of stride between kernels
        :param padding: number of padding pixels around the data
        """
        pool_layer = gl.deeplearning.layers.AveragePoolingLayer(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.layers.append(pool_layer)

    def add_flatten_layer(self):
        """Add a flattening layer to the neural net

        This method must be called before adding a fully connected layer
        """
        self.layers.append(
            gl.deeplearning.layers.FlattenLayer()
        )

    def add_full_connection_layer(self, num_hidden_units,
                                  init_bias=0, init_sigma=0.01,
                                  init_random='gaussian'):
        """

        :param num_hidden_units:
        :param init_bias:
        :param init_sigma:
        :param init_random:
        :return:
        """
        fc_layer = gl.deeplearning.layers.FullConnectionLayer(
            num_hidden_units, init_bias=init_bias, init_sigma=init_sigma,
            init_random=init_random
        )
        self.layers.append(fc_layer)
        return

    def add_relu_layer(self):
        """Add a rectified linear activation unit to the network
        """
        self.layers.append(gl.deeplearning.layers.RectifiedLinearLayer())
        return

    def add_sigmoid_layer(self):
        """Add a sigmoid activation layer to the network
        """
        self.layers.append(gl.deeplearning.layers.SigmoidLayer())

    def add_tanh_layer(self):
        """Add a hyperbolic tangent layer to the network
        """
        self.layers.append(gl.deeplearning.layers.TanhLayer())

    def add_soft_plus_layer(self):
        """Add a softplus activation layer to the network
        """
        self.layers.append(gl.deeplearning.layers.SoftplusLayer())

    def add_soft_max_layer(self):
        """Add a softmax layer to the network
        """
        self.layers.append(gl.deeplearning.layers.SoftmaxLayer())

    def add_dropout_layer(self, threshold=0.5):
        """Add a dropout layer to the network

        Use this layer to avoid overfitting

        :param threshold: probability of setting neuron to 0
        :return:
        """
        drop_layer = gl.deeplearning.layers.DropoutLayer(threshold)
        self.layers.append(drop_layer)

    def __getitem__(self, item):
        return self.net.params[item]

    def __setitem__(self, key, value):
        self.net.params[key] = value

    def set_params_from_file(self, path):
        """Set the general network parameters from text file

        Format is: key, val, [i,f,b,'']
        for integer, float, binary and string values
        """
        with open(path) as f:
            for line in f:
                key_val = line.split(',')
                if len(key_val) == 3:
                    key = key_val[0].strip()
                    value = key_val[1].strip()
                    type = key_val[2].strip()
                    if type == 'i':
                        value = int(value)
                    if type == 'f':
                        value = float(value)
                    if type == 'b':
                        if value.lower() == 'false':
                            value = False
                        else:
                            value = True
                    self.net.params[key] = value
