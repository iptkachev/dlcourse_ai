class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        self._zero_grad()
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        predictions = self._forward(X)
        loss, dprediction = softmax_with_cross_entropy(predictions, y)
        self._backward(dprediction)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        loss_l2 = self._backward_l2_regularization()
        loss += loss_l2
        
        return loss

    def _zero_grad(self):
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
                
    def _forward(self, X):
        output = self.fc1.forward(X)
        output = self.relu1.forward(output)
        output = self.fc2.forward(output)
        return output
    
    def _backward(self, dprediction):
        grad = self.fc2.backward(dprediction)
        grad = self.relu1.backward(grad)
        self.fc1.backward(grad)
        
    def _backward_l2_regularization(self):
        loss_l2 = 0.
        
        for param in self.params().values():
            param_loss_l2, param_grad_l2 = l2_regularization(param.value, self.reg)
            param.grad += param_grad_l2
            loss_l2 += param_loss_l2
                
        return loss_l2
    
    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        predictions = self._forward(X)
        y_pred = np.argmax(softmax(predictions), axis=1)

        return y_pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        for k, v in self.fc2.params().items():
            result["".join(["fc2_", k])] = v
        
        for k, v in self.fc1.params().items():
            result["".join(["fc1_", k])] = v
            

        
        return result