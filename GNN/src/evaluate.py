import numpy as np

try:
    import torch
except ImportError:
    torch = None


class Evaluator:
    """Minimal RMSE evaluator for single-task regression."""

    def __init__(self, name: str):
        self.name = name
        self.num_tasks = 1
        self.eval_metric = 'rmse'

    def _to_numpy_2d(self, x):
        if torch is not None and isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if not isinstance(x, np.ndarray):
            raise RuntimeError('inputs must be numpy arrays or torch tensors')
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise RuntimeError(f'inputs must be 1D or 2D arrays, got {x.ndim}D')
        return x

    def eval(self, input_dict):
        if 'y_true' not in input_dict or 'y_pred' not in input_dict:
            raise RuntimeError("input_dict must contain 'y_true' and 'y_pred'")

        y_true = self._to_numpy_2d(input_dict['y_true'])
        y_pred = self._to_numpy_2d(input_dict['y_pred'])

        if y_true.shape != y_pred.shape:
            raise RuntimeError('Shape of y_true and y_pred must be the same')
        if y_true.shape[1] != self.num_tasks:
            raise RuntimeError(f'Number of tasks for {self.name} should be {self.num_tasks} but {y_true.shape[1]} given')

        return self._eval_rmse(y_true, y_pred)

    # Extra docs and formats removed for brevity.

    def _eval_rmse(self, y_true, y_pred):
        """Compute RMSE averaged across tasks, ignoring nans in y_true."""
        rmse_list = []
        is_labeled = y_true == y_true
        for i in range(y_true.shape[1]):
            y_true_task = y_true[is_labeled[:, i], i]
            y_pred_task = y_pred[is_labeled[:, i], i]
            rmse = np.sqrt(((y_true_task - y_pred_task) ** 2).mean())
            rmse_list.append(rmse)
        return {'rmse': float(sum(rmse_list) / len(rmse_list))}

