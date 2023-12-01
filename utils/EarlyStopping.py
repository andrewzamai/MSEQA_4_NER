import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode="min", save_best_weights=True):
        """
        patience : int [default=10]
            Number of epochs without improvement to wait before stopping the training.
        min_delta : float [default=0]
            Minimum value to identify as an improvement.
        mode : str [default="min"]
            One of "min" or "max". Identify whether an improvement
            consists on a smaller or bigger value in the metric.
        save_best_weights : bool [default=True]
            Whether to save the model state when there is an improvement or not.
        """
        # save initialization argument
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        # determine which is the definition of better score given the mode
        if self.mode == "min":
            self._is_better = lambda new, old: new < old - self.min_delta
            self.best_score = np.Inf
        elif self.mode == "max":
            self._is_better = lambda new, old: new > old + self.min_delta
            self.best_score = -np.Inf

        self.save_best_weights = save_best_weights
        self.best_weights = None
        # keep tracks of number of iterations without improvements
        self.n_iter_no_improvements = 0
        # whether to stop the training or not
        self.stop_training = False

    def step(self, metric, model=None):
        # if there is an improvements, update `best_score` and save model weights
        if self._is_better(metric, self.best_score):
            self.best_score = metric
            self.n_iter_no_improvements = 0
            if self.save_best_weights and model is not None:
                self.best_weights = model.state_dict()
        # otherwise update counter
        else:
            self.n_iter_no_improvements += 1

        # if no improvements for more epochs than patient, stop the training
        # (set the flag to False)
        if self.n_iter_no_improvements >= self.patience:
            self.stop_training = True
            print(f"Early Stopping: monitored quantity did not improved in the last {self.patience} epochs.")

        return self.stop_training
