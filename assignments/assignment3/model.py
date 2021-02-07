import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        image_width, image_height, n_channels = input_shape
        conv_padding = 0
        conv_filter_size = 3
        max_pool_size = 4
        max_pool_stride = 1
        
        conv1_output_size = image_width - conv_filter_size + 1
        maxpool1_output_size = int((conv1_output_size - max_pool_size) / max_pool_stride) + 1
        conv2_output_size = maxpool1_output_size - conv_filter_size + 1
        maxpool2_output_size = int((conv2_output_size - max_pool_size) / max_pool_stride) + 1
        # correct if height == width !!!
        fc_input_size = maxpool2_output_size * maxpool2_output_size * conv2_channels  

        self.conv1 = ConvolutionalLayer(n_channels, conv1_channels, conv_filter_size, conv_padding)
        self.relu1 = ReLULayer()
        self.maxpool1 = MaxPoolingLayer(max_pool_size, max_pool_stride)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, conv_filter_size, conv_padding)
        self.relu2 = ReLULayer()
        self.maxpool2 = MaxPoolingLayer(max_pool_size, max_pool_stride)
        self.flattener = Flattener()
        self.fc = FullyConnectedLayer(fc_input_size, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        self._zero_grad()
        
        predictions = self._forward(X)
        loss, dprediction = softmax_with_cross_entropy(predictions, y)
        self._backward(dprediction)
        
        return loss

    def _zero_grad(self):
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
                
    def _forward(self, X):
        output = self.conv1.forward(X)
        output = self.relu1.forward(output)
        output = self.maxpool1.forward(output)
        output = self.conv2.forward(output)
        output = self.relu2.forward(output)
        output = self.maxpool2.forward(output)
        output = self.flattener.forward(output)
        output = self.fc.forward(output)
        return output
    
    def _backward(self, dprediction):
        grad = self.fc.backward(dprediction)
        grad = self.flattener.backward(grad)
        grad = self.maxpool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.maxpool1.backward(grad)
        grad = self.relu1.backward(grad)
        self.conv1.backward(grad)
        
    def predict(self, X):
        # You can probably copy the code from previous assignment
        predictions = self._forward(X)
        y_pred = np.argmax(softmax(predictions), axis=1)

        return y_pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        for k, v in self.conv1.params().items():
            result["".join(["conv1_", k])] = v 
        for k, v in self.conv2.params().items():
            result["".join(["conv2_", k])] = v
        for k, v in self.fc.params().items():
            result["".join(["fc_", k])] = v
        
        return result
