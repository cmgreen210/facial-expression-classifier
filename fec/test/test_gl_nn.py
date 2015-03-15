import unittest
from fec.classifier.gl_nn import GraphLabNeuralNetBuilder
import os


class GraphLabNeuralNetTest(unittest.TestCase):

    def test_convolution_layer(self):
        nn = GraphLabNeuralNetBuilder()
        nn.add_convolution_layer(3, 1, 10)

        nn.add_convolution_layer(3, 1, 10, **{'padding': 1})

        return

    def test_max_pooling_layer(self):
        nn = GraphLabNeuralNetBuilder()
        nn.add_max_pooling_layer(3)
        nn.add_max_pooling_layer(3, padding=0)
        nn.add_max_pooling_layer(3, 4, 5)

    def test_avg_pooling_layer(self):
        nn = GraphLabNeuralNetBuilder()
        nn.add_avg_pooling_layer(3)
        nn.add_avg_pooling_layer(3, padding=0)
        nn.add_avg_pooling_layer(3, 4, 5)

    def test_flatten_layer(self):
        nn = GraphLabNeuralNetBuilder()
        nn.add_flatten_layer()

    def test_full_connection_layer(self):
        nn = GraphLabNeuralNetBuilder()
        nn.add_flatten_layer()
        nn.add_full_connection_layer(10)
        nn.add_full_connection_layer(100)

    def test_activation_layers(self):
        nn = GraphLabNeuralNetBuilder()
        nn.add_relu_layer()
        nn.add_sigmoid_layer()
        nn.add_tanh_layer()
        nn.add_soft_max_layer()

    def test_drop_out_layer(self):
        nn = GraphLabNeuralNetBuilder()
        nn.add_dropout_layer()
        nn.add_dropout_layer(.8)

    def test_get_net_bad(self):
        nn_bad = GraphLabNeuralNetBuilder()
        self.assertRaises(ValueError, nn_bad.get_net)  # Empty net

    def test_net_params(self):
        nn = GraphLabNeuralNetBuilder()
        nn['momentum'] = 0.5
        self.assertEqual(nn['momentum'], 0.5)

    def test_set_params_from_file(self):
        dir, _ = os.path.split(os.path.abspath(__file__))
        file_path = os.path.join(dir, 'data', 'net_params.txt')

        net = GraphLabNeuralNetBuilder()
        net.set_params_from_file(file_path)
        self.assertEqual(net['learning_rate'], 0.1)
        self.assertEqual(net['batch_size'], 256)
        self.assertEqual(net['learning_rate_schedule'], 'constant')
        self.assertEqual(net['subtract_mean'], False)
        print net.net
