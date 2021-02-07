import numpy as np
import sys
sys.path.append('..')
from assignment1.linear_classifer import softmax_with_cross_entropy, softmax, l2_regularization
from assignment2.layers import Param, ReLULayer, FullyConnectedLayer


class ConvolutionalLayer:
    BATCH_INDEX = 0
    HEIGHT_INDEX = 1
    WIDTH_INDEX = 2
    
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.X_padded = None
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, _ = X.shape

        out_height = (height - self.filter_size + 2 * self.padding) + 1
        out_width = (width - self.filter_size + 2 * self.padding) + 1
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        result = np.zeros(
            (batch_size, out_height, out_width, self.out_channels)
        )

        self.X_padded = self._pad_X(X)  # returns new object
        W_reshaped = self.W.value.reshape(-1, self.out_channels)        

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            slice_height = slice(y, (y + self.filter_size))
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                slice_width = slice(x, (x + self.filter_size))
                X_region = self.X_padded[:, slice_height, slice_width, :].reshape(batch_size, -1)            
                result[:, y, x, :] = X_region @ W_reshaped + self.B.value
                
        return result
    
    def _pad_X(self, X):
        return np.pad(X, ((0,), (self.padding,), (self.padding,), (0,)), 'constant', constant_values=0)
    
    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, _ = self.X_shape
        _, out_height, out_width, _ = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        d_result = np.zeros(
            (batch_size, height + 2 * self.padding, width + 2 * self.padding, self.in_channels)
        )

        W_reshaped = self.W.value.reshape(-1, self.out_channels)
        # Try to avoid having any other loops here too
        for y in range(out_height):
            slice_height = slice(y, (y + self.filter_size))
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                slice_width = slice(x, (x + self.filter_size))
                d_out_region = d_out[:, y, x, :]
                
                d_result_region = (d_out_region @ W_reshaped.T).reshape(
                    batch_size, self.filter_size, self.filter_size, self.in_channels
                )
                d_result[:, slice_height, slice_width, :] += d_result_region
                
                X_region = self.X_padded[:, slice_height, slice_width, :].reshape(batch_size, -1)
                W_grad_region = (X_region.T @ d_out_region.reshape(batch_size, -1)).reshape(self.W.grad.shape)
                self.W.grad += W_grad_region
                self.B.grad += d_out_region.sum(axis=self.BATCH_INDEX)
                
        height_without_padding = slice(self.padding, (height + self.padding))
        width_without_padding = slice(self.padding, (width + self.padding))  
        
        return d_result[:, height_without_padding, width_without_padding, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    HEIGHT_INDEX = 1
    WIDTH_INDEX = 2
    
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        
        out_height = int((height - self.pool_size) / self.stride + 1)
        out_width = int((width - self.pool_size) / self.stride + 1)

        result = np.zeros(
            (batch_size, out_height, out_width, channels)
        )
        
        for y in range(0, out_height, self.stride):
            slice_height = slice(y, (y + self.pool_size))
            for x in range(0, out_width, self.stride):
                # TODO: Implement forward pass for specific location
                slice_width = slice(x, (x + self.pool_size))
                result[:, y, x, :] = self.X[:, slice_height, slice_width, :].max(
                    axis=(self.HEIGHT_INDEX, self.WIDTH_INDEX)
                )
                
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        
        d_result = np.zeros(
            (batch_size, height, width, channels)
        )

        # Try to avoid having any other loops here too
        for y in range(0, out_height, self.stride):
            slice_height = slice(y, (y + self.pool_size))
            for x in range(0, out_width, self.stride):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                slice_width = slice(x, (x + self.pool_size))
                d_out_region = d_out[:, [[y]], [[x]], :]  # hack to add axis for right broadcast
                X_region_grad = self._grad_MaxPool(self.X[:, slice_height, slice_width, :])
                d_result_region = (d_out_region * X_region_grad)
                d_result[:, slice_height, slice_width, :] += d_result_region
                
        return d_result

    def _grad_MaxPool(self, X_region):
        X_region_max = X_region.max(
            axis=(self.HEIGHT_INDEX, self.WIDTH_INDEX), keepdims=True
        )
        return (X_region == X_region_max).astype('int32')
    
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, *_ = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
