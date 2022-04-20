import numpy as np


class MeanSquaredError:
    def _loss(self, y_pred, y_true):
        print(((y_pred - y_true)))
        return ((y_pred - y_true) ** 2).mean()

    def _loss_gradient(self, y_pred, y_true):
        return 2 * (y_pred - y_true)


class MeanAbsolutePercentageError:
    def _loss(self, y_pred, y_true):
        _y_true = y_true + 1e-7
        return np.abs((y_pred - _y_true) / _y_true).mean()

    def _loss_gradient(self, y_pred, y_true):
        _y_true = y_true + 1e-7
        less_than_mask = y_pred < _y_true
        greater_than_mask = y_pred > _y_true
        return (1 / (_y_true.size * _y_true)) * greater_than_mask - (
            1 / (_y_true.size * _y_true)
        ) * less_than_mask


LOSS_FUNCTIONS = {
    "mse": MeanSquaredError,
    "mape": MeanAbsolutePercentageError,
}
