import numpy as np
from scipy.special import expit

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_nodes, output_size, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        prev_size = input_size
        # Initialize hidden layers
        for _ in range(hidden_layers):
            self.weights.append(np.random.randn(prev_size, hidden_nodes))  # He initialization
            self.biases.append(np.zeros((1, hidden_nodes)))
            prev_size = hidden_nodes
        # Output layer
        self.weights.append(np.random.randn(prev_size, output_size))
        self.biases.append(np.zeros((1, output_size)))
    

    @staticmethod
    def __sigmoid(x):
        return expit(x)
    

    @staticmethod
    def __sigmoid_derivative(sig):
        return sig * (1 - sig)
    

    def __forward(self, X):
        activations = [X]  # Store input and layer outputs
        for w, b in zip(self.weights, self.biases):
            z = activations[-1] @ w + b
            a = self.__sigmoid(z)
            activations.append(a)
        return activations
    

    def __backward(self, activations, y):
        output_error = (y - activations[-1])
        delta = output_error * self.__sigmoid_derivative(activations[-1])
        deltas = [delta]
        
        # Propagate error backwards
        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1] @ self.weights[i+1].T
            delta *= self.__sigmoid_derivative(activations[i+1])  # Current layer's activation
            deltas.append(delta)
        
        deltas.reverse()  # Order now: [input_layer_delta, hidden_deltas..., output_delta]
        
        grad_weights = []
        grad_biases = []
        # Calculate gradients for each layer
        for i in range(len(self.weights)):
            grad_w = activations[i].T @ deltas[i]  # Input to layer i
            grad_b = np.sum(deltas[i], axis=0, keepdims=True)
            grad_weights.append(grad_w)
            grad_biases.append(grad_b)
        
        return grad_weights, grad_biases
    

    def train(self, X, y, epochs=1, batch_size=32):
        num_batches = max(len(X) // batch_size, 1)
        indices = np.arange(len(X))
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            X_batches = np.array_split(X_shuffled, num_batches)
            y_batches = np.array_split(y_shuffled, num_batches)
            
            for X_batch, y_batch in zip(X_batches, y_batches):
                grad_weights_sum = [np.zeros_like(w) for w in self.weights]
                grad_biases_sum = [np.zeros_like(b) for b in self.biases]
                
                for x_i, y_i in zip(X_batch, y_batch):
                    x_i = x_i.reshape(1, -1)
                    y_i = y_i.reshape(1, -1)
                    
                    activations = self.__forward(x_i)
                    grad_weights, grad_biases = self.__backward(activations, y_i)
                    
                    for layer in range(len(self.weights)):
                        grad_weights_sum[layer] += grad_weights[layer]
                        grad_biases_sum[layer] += grad_biases[layer]
                
                # Update parameters with average gradients
                batch_len = len(X_batch)
                for layer in range(len(self.weights)):
                    self.weights[layer] += self.learning_rate * (grad_weights_sum[layer] / batch_len)
                    self.biases[layer] += self.learning_rate * (grad_biases_sum[layer] / batch_len)
            
            # Calculate loss
            if epoch % 2 == 0:
                pred = self.__forward(X)[-1]
                loss = np.mean((pred - y)**2)
                print(f"Epoch {epoch} - Loss: {loss:.4f}")
    

    def predict(self, X):
        X = X.reshape(1, -1) if X.ndim == 1 else X
        return self.__forward(X)[-1] > 0.5