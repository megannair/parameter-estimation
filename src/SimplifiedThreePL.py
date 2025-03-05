import numpy as np
from scipy.optimize import minimize

class SimplifiedThreePL:
    def __init__(self, experiment):
        """
        Initializes the SimplifiedThreePL model.
        
        Args:
            experiment: An instance of the Experiment class.
        """
        self.experiment = experiment
        self._base_rate = None
        self._logit_base_rate = None
        self._discrimination = None
        self._is_fitted = False

    def summary(self):
        """Returns a summary dictionary of the experiment data."""
        return self.experiment.summary()

    def predict(self, parameters):
        """Computes the probability of a correct response for each condition."""
        a, q = parameters
        c = 1 / (1 + np.exp(-q))  # Convert logit base rate to c
        b = np.array([2, 1, 0, -1, -2])  # Given difficulty values
        theta = 0  # Fixed ability parameter

        prob_correct = c + (1 - c) * (1 / (1 + np.exp(-a * (theta - b))))
        return prob_correct

    def negative_log_likelihood(self, parameters):
        """Computes the negative log-likelihood of the data given the parameters."""
        prob_correct = self.predict(parameters)
        log_likelihood = np.sum(
            self.experiment.n_correct * np.log(prob_correct) +
            self.experiment.n_incorrect * np.log(1 - prob_correct)
        )
        return -log_likelihood  # We minimize negative log-likelihood

    def fit(self):
        """Finds the best-fitting parameters using maximum likelihood estimation."""
        initial_guess = [1.0, 0.0]  # Initial guess for a and q
        
        result = minimize(self.negative_log_likelihood, initial_guess, method='L-BFGS-B')

        if result.success:
            self._discrimination, self._logit_base_rate = result.x
            self._base_rate = 1 / (1 + np.exp(-self._logit_base_rate))
            self._is_fitted = True
        else:
            raise RuntimeError("Optimization failed to converge.")

    def get_discrimination(self):
        """Returns the estimated discrimination parameter a."""
        if not self._is_fitted:
            raise ValueError("Model is not yet fitted.")
        return self._discrimination

    def get_base_rate(self):
        """Returns the estimated base rate parameter c."""
        if not self._is_fitted:
            raise ValueError("Model is not yet fitted.")
        return self._base_rate
