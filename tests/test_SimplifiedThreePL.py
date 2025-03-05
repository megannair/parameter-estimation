import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment import Experiment

class TestSimplifiedThreePL(unittest.TestCase):
    
    def setUp(self):
        """Set up an example experiment."""
        self.n_correct = np.array([55, 60, 75, 90, 95])
        self.n_incorrect = 100 - self.n_correct
        self.experiment = Experiment(self.n_correct, self.n_incorrect)
        self.model = SimplifiedThreePL(self.experiment)

    def test_initialization(self):
        """Test model initialization."""
        summary = self.model.summary()
        self.assertEqual(summary["n_conditions"], 5)

    def test_predict(self):
        """Test prediction values are between 0 and 1."""
        params = [1.0, 0.0]
        predictions = self.model.predict(params)
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))

    def test_fit(self):
        """Test that model fitting works."""
        self.model.fit()
        self.assertTrue(self.model._is_fitted)
        self.assertGreater(self.model.get_discrimination(), 0)
        self.assertGreater(self.model.get_base_rate(), 0)

if __name__ == "__main__":
    unittest.main()
