"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict

from typing import Callable, List, Literal, Tuple, Union


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )

    elif name == "pool2d":
        return Pool2D(kernel_shape=kernel_shape, mode=mode, stride=stride, pad=pad)

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights([self.n_in, self.n_out])
        b = np.zeros((1, self.n_out))
        
        dW = np.zeros((self.n_in, self.n_out))
        db = np.zeros((1, self.n_out))
        
        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache: OrderedDict = OrderedDict({})  # cache for backprop
        self.gradients: OrderedDict = OrderedDict({"W": dW, "b": db})  # parameter gradients initialized to zero
                                           # MUST HAVE THE SAME KEYS AS `self.parameters`

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###
        
        # perform an affine transformation and activation
        Z = np.matmul(X, self.parameters["W"]) + self.parameters["b"]
        out = self.activation.forward(Z)
        
        # store information necessary for backprop in `self.cache`
        k = len(self.cache.keys())
        self.cache[k + 1] = Z
        self.cache[k + 2] = X

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  derivative of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        
        # unpack the cache
        X = self.cache.popitem()[1]
        Z = self.cache.popitem()[1]
        
        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer

        dLdZ = self.activation.backward(Z, dLdY)

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.
        
        self.gradients["W"] = np.matmul(X.T, dLdZ)
        self.gradients["b"] = np.mean(dLdZ, axis = 0)
        dX = np.matmul(dLdZ, self.parameters["W"].T)

        ### END YOUR CODE ###

        return dX


class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache = OrderedDict({"Z": [], "X": []})
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)})

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)
        
        ### BEGIN YOUR CODE ###
        
        # implement a convolutional forward pass
        
        out_rows = (in_rows - kernel_height + 2 * self.pad[0]) // self.stride + 1
        out_cols = (in_cols - kernel_width + 2 * self.pad[1]) // self.stride + 1
        
        X = np.pad(X, ((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)))
        
        Z = np.zeros((n_examples, out_rows, out_cols, out_channels))
        # For each u, v element in H
        for u in range(out_rows):
            for v in range(out_cols):

                # Perform Sum
                Zuv = np.zeros((n_examples, out_channels))
                for i in range(kernel_height):
                    for j in range(kernel_width):
                        Zuv += np.matmul(X[:, i + u * self.stride, j + v * self.stride, :], W[i, j, :, :])
                Zuv += b
                Z[:, u, v, :] = Zuv
        
        # cache any values required for backprop
        self.cache["Z"].append(Z)
        self.cache["X"].append(X)
        
        H = self.activation.forward(Z)
        out = H
        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  derivative of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        
        X = self.cache["X"].pop()
        Z = self.cache["Z"].pop()
        W = self.parameters["W"]
        _, r1, r2, _ = Z.shape
        k1, k2, _, _ = W.shape
        n_examples = X.shape[0]
        
        
        # perform a backward pass
        dX = np.zeros(X.shape)
        dW = np.zeros(W.shape)
        
        dZ = self.activation.backward(Z, dLdY) # times dLdY
        
        for i in range(k1):
            for j in range(k2):
                dWij = np.zeros((W.shape[2], W.shape[3]))
                for u in range(r1):
                    for v in range(r2):
                        dLdZuv = dZ[:, u, v, :]
                        dX[:, i + u * self.stride, j + v * self.stride, :] += np.matmul(dLdZuv, W[i, j, :, :].T)
                        Xuvij = X[:, i + u * self.stride, j + v * self.stride, :]
                        dWij += np.matmul(Xuvij.T, dLdZuv)
                dW[i, j, :, :] = dWij
                
        #self.gradients["W"] = dW
        
        db = np.zeros((1, self.n_out))
        
        for u in range(r1):
            for v in range(r2):
                db += np.sum(dZ[:, u, v, :], axis = 0)
            
        ### END YOUR CODE ###

        return dX[:, self.pad[0]:-self.pad[0], self.pad[1]:-self.pad[1], :]

class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        ### BEGIN YOUR CODE ###
        
        n_examples, in_rows, in_cols, channels = X.shape
        t = self.stride
        
        # implement the forward pass
        kernel_height, kernel_width = self.kernel_shape
        out_rows = (in_rows - kernel_height + 2 * self.pad[0]) // self.stride + 1
        out_cols = (in_cols - kernel_width + 2 * self.pad[1]) // self.stride + 1
        
        X = np.pad(X, ((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)))
        
        Z = np.zeros((n_examples, out_rows, out_cols, channels))
        
        for u in range(out_rows):
            for v in range(out_cols):
                tu1 = t * u
                tu2 = t * u + kernel_height
                tv1 = t * v
                tv2 = t * v + kernel_width

                Z[:, u, v, :] = self.pool_fn(X[:, tu1:tu2, tv1:tv2, :], axis = (1, 2))
        
        # cache any values required for backprop
        self.cache["X_pad"].append(X)        
        
        X_pool = Z
    
        ### END YOUR CODE ###

        return X_pool

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """
        ### BEGIN YOUR CODE ###
        
        if self.mode == "max":
            
            X = self.cache["X_pad"].pop()
            kernel_height, kernel_width = self.kernel_shape
            _, out_rows, out_cols, channels = dLdY.shape
            t, p = self.stride, self.pad
            dX = np.zeros(X.shape)
            
            for u in range(out_rows):
                for v in range(out_cols):
                    for c in range(channels):
                        tu1 = t * u
                        tu2 = t * u + kernel_height
                        tv1 = t * v
                        tv2 = t * v + kernel_width

                        Xuv = X[:, tu1:tu2, tv1:tv2, c]
                        a012 = np.arange(Xuv.shape[0])
                        Xuvr = Xuv.reshape(X.shape[0], -1)
                        am = self.arg_pool_fn(Xuvr, axis = 1)
                        ui = np.unravel_index(am, Xuv[0].shape)
                        idx = (a012, ui[0] + tu1, ui[1] + tv1, c)
                        dX[idx] += dLdY[:, u, v, c]
        
        if self.mode == "average":
            
            X = self.cache["X_pad"].pop()
            kernel_height, kernel_width = self.kernel_shape
            _, out_rows, out_cols, channels = dLdY.shape
            t, p = self.stride, self.pad
            dX = np.zeros(X.shape)
            
            for u in range(out_rows):
                for v in range(out_cols):
                    
                    tu1 = t * u
                    tu2 = t * u + kernel_height
                    tv1 = t * v
                    tv2 = t * v + kernel_width
                        
                    for c in range(channels):
                        
                        dLuv = dLdY[:, u, v, c] / (kernel_height * kernel_width)
                        dX[:, tu1:tu2, tv1:tv2, c] += dLuv.reshape((dLuv.shape[0], 1, 1))
        
        ### END YOUR CODE ###
        
        p0e = dX.shape[1] - p[0]
        p1e = dX.shape[2] - p[1]

        return dX[:, p[0]:p0e, p[1]:p1e, :] 

class Flatten(Layer):
    """Flatten the input array."""

    def __init__(self, keep_dim: str = "first") -> None:
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        in_dims = self.cache["in_dims"]
        dX = dLdY.reshape(in_dims)
        return dX
