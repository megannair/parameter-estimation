import numpy as np

class Experiment:
    def __init__(self, n_correct, n_incorrect):
        """
        Initializes an Experiment with observed correct and incorrect responses.
        
        Args:
            n_correct (array-like): Number of correct responses in each condition.
            n_incorrect (array-like): Number of incorrect responses in each condition.
        """
        self.n_correct = np.array(n_correct)
        self.n_incorrect = np.array(n_incorrect)

        if len(self.n_correct) != len(self.n_incorrect):
            raise ValueError("n_correct and n_incorrect must have the same length.")

    def summary(self):
        """Returns a summary of the experiment data."""
        return {
            "n_total": np.sum(self.n_correct + self.n_incorrect),
            "n_correct": np.sum(self.n_correct),
            "n_incorrect": np.sum(self.n_incorrect),
            "n_conditions": len(self.n_correct)
        }

