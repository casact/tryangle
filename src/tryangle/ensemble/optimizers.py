# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent optimizer with decay and momentum

    Parameters
    ----------
    learning_rate : float, default=0.0001
        The initial learning rate used. It controls the step-size used
        in updating the weights and biases.

    decay : float, default=0.0
        The learning rate decay over epochs.

    momentum : float, default=0.0
        The value of momentum used. Must be larger than or equal to 0.
    """

    def __init__(self, learning_rate=0.0001, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self._learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum

    def reset(self, model):
        model.weight_momentums = np.zeros_like(model.initial_weight)
        model.bias_momentums = np.zeros_like(model.initial_bias)

    def pre_update_params(self, epoch):
        if self.decay:
            self._learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * epoch)
            )

    def update_params(self, model, epoch):
        if self.momentum:
            if not hasattr(model, "weight_momentums"):
                self.reset(model)

            weight_updates = (
                self.momentum * model.weight_momentums
                - self._learning_rate * model._w_grad
            )
            model.weight_momentums = weight_updates

            bias_updates = (
                self.momentum * model.bias_momentums
                - self._learning_rate * model._b_grad
            )
            model.bias_momentums = bias_updates
        else:
            weight_updates = -self._learning_rate * model._w_grad
            bias_updates = -self._learning_rate * model._b_grad

        model.weights = model.weights + weight_updates
        model.biases = model.biases + bias_updates


class AdaGrad:
    """
    Adaptive Gradient Algorithm with decay

    Parameters
    ----------
    learning_rate : float, default=0.0001
        The initial learning rate used. It controls the step-size used
        in updating the weights and biases.

    decay : float, default=0.0
        The learning rate decay over epochs.

    epsilon : float, default=1e-7
        A small constant for numerical stability.
    """

    def __init__(self, learning_rate=0.0001, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self._learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon

    def pre_update_params(self, epoch):
        if self.decay:
            self._learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * epoch)
            )

    def update_params(self, model, epoch):
        if not hasattr(model, "weight_cache"):
            model.weight_cache = np.zeros_like(model.weights)
            model.bias_cache = np.zeros_like(model.biases)

        model.weight_cache += model._w_grad**2
        model.bias_cache += model._b_grad**2

        model.weights += (
            -model.optimizer._learning_rate
            * model._w_grad
            / (np.sqrt(model.weight_cache) + self.epsilon)
        )
        model.biases += (
            -model.optimizer._learning_rate
            * model._w_grad
            / (np.sqrt(model.bias_cache) + self.epsilon)
        )


class RMSProp:
    """
    RMSProp with decay

    Parameters
    ----------
    learning_rate : float, default=0.0001
        The initial learning rate used. It controls the step-size used
        in updating the weights and biases.

    decay : float, default=0.0
        The learning rate decay over epochs.

    epsilon : float, default=1e-7
        A small constant for numerical stability.

    rho : float, default=0.9
        Discounting factor for the history/coming gradient.
    """

    def __init__(self, learning_rate=0.0001, decay=0.0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self._learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho

    def reset(self, model):
        model.weight_cache = np.zeros_like(model.weights)
        model.bias_cache = np.zeros_like(model.biases)

    def pre_update_params(self, epoch):
        if self.decay:
            self._learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * epoch)
            )

    def update_params(self, model, epoch):
        if not hasattr(model, "weight_cache"):
            self.reset(model)

        model.weight_cache += (
            self.rho * model.weight_cache + (1 - self.rho) * model._w_grad**2
        )
        model.bias_cache += (
            self.rho * model.bias_cache + (1 - self.rho) * model._b_grad**2
        )

        model.weights += (
            -model.optimizer._learning_rate
            * model._w_grad
            / (np.sqrt(model.weight_cache) + self.epsilon)
        )
        model.biases += (
            -model.optimizer._learning_rate
            * model._w_grad
            / (np.sqrt(model.bias_cache) + self.epsilon)
        )


class Adam:
    """
    Adam optimizer with decay

    Parameters
    ----------
    learning_rate : float, default=0.0001
        The initial learning rate used. It controls the step-size used
        in updating the weights and biases.

    decay : float, default=0.0
        The learning rate decay over epochs.

    epsilon : float, default=1e-7
        A small constant for numerical stability.

    beta_1 : float, default=0.9
        Exponential decay rate for the 1st moment estimates.

    beta_2 : float, default=0.999
        Exponential decay rate for the 2nd moment estimates.
    """

    def __init__(
        self, learning_rate=0.0001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999
    ):

        self.learning_rate = learning_rate
        self._learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def reset(self, model):
        model.weight_momentums = np.zeros_like(model.weights)
        model.weight_cache = np.zeros_like(model.weights)
        model.bias_momentums = np.zeros_like(model.biases)
        model.bias_cache = np.zeros_like(model.biases)

    def pre_update_params(self, epoch):
        if self.decay:
            self._learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * epoch)
            )

    def update_params(self, model, epoch):
        if not hasattr(model, "weight_cache"):
            self.reset(model)

        model.weight_momentums = (
            self.beta_1 * model.weight_momentums + (1 - self.beta_1) * model._w_grad
        )
        model.bias_momentums = (
            self.beta_1 * model.bias_momentums + (1 - self.beta_1) * model._b_grad
        )

        weight_momentums_corrected = model.weight_momentums / (
            1 - self.beta_1 ** (epoch + 1)
        )
        bias_momentums_corrected = model.bias_momentums / (
            1 - self.beta_1 ** (epoch + 1)
        )

        model.weight_cache += (
            self.beta_2 * model.weight_cache + (1 - self.beta_2) * model._w_grad**2
        )
        model.bias_cache += (
            self.beta_2 * model.bias_cache + (1 - self.beta_2) * model._b_grad**2
        )

        weight_cache_corrected = model.weight_cache / (1 - self.beta_2 ** (epoch + 1))
        bias_cache_corrected = model.bias_cache / (1 - self.beta_2 ** (epoch + 1))

        model.weights += (
            -model.optimizer._learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        model.biases += (
            -model.optimizer._learning_rate
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )
