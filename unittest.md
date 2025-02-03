# Unittest for testing the functions 

#### Import all libraries and the bayesian.py to test
If you've saved your bayesian_test.py in your directory, import it with the below
~~~python
from bayesian_test import BayesianMultivariateTest
import unittest
import pandas as pd
import numpy as np
~~~

##### setUp()
The first function setUp() initializes a mock dataset to test the functions based on the mock data

##### test_validate_data()

Makes sure that we have the correct number of combinations from our mock dataset

##### test_calculate_posteriors()
This method asserts we have the right number of combinations from our mock data and also makes sure that our posterior mean is greater than 0

##### test_probability_of_being_best()
Ensures that our posterior probabilities i.e probability of being best sums up to 1

##### test_update_prior_with_posterior()
Makes sure the posterior updates correctly from the prior

##### test_estimate_days_to_run_exp()
Ensures that the number of estimated days to run the experiment returns an integer and also greater than 0 based on the average daily visitors and threshold of 95%


~~~python
from All_classes import BayesianMultivariateTest
import unittest
import pandas as pd
import numpy as np


class TestBayesianMultivariateTest(unittest.TestCase):
    def setUp(self):
        """Set up a mock dataset for testing."""
        self.test_data = pd.DataFrame({
            'Combination': ['A', 'B', 'C','D'],
            'Conversions': [50, 60, 55, 49],
            'Visitors': [1000, 1000, 1000, 1000]
        })
        self.test_data.to_csv("test_data.csv", index=False)
        self.bayesian_test = BayesianMultivariateTest("test_data.csv")

    def test_validate_data(self):
        """Test data validation method."""
        self.assertEqual(len(self.bayesian_test.data), 4)
        self.assertIn('Combination', self.bayesian_test.data.columns)

    def test_calculate_posteriors(self):
        """Test posterior calculations."""
        posteriors = self.bayesian_test.calculate_posteriors()
        self.assertEqual(len(posteriors), 4)
        self.assertTrue(all(posterior.mean() > 0 for posterior in posteriors.values()))

    def test_probability_of_being_best(self):
        """Test probability of being the best."""
        self.bayesian_test.calculate_posteriors()
        probabilities = self.bayesian_test.probability_of_being_best()
        self.assertAlmostEqual(probabilities.sum(), 1, places=2)

    def test_update_prior_with_posterior(self):
        """Test that prior updates correctly."""
        self.bayesian_test.calculate_posteriors()
        old_alpha, old_beta = self.bayesian_test.prior_alpha, self.bayesian_test.prior_beta
        self.bayesian_test.update_prior_with_posterior()
        self.assertNotEqual((old_alpha, old_beta), (self.bayesian_test.prior_alpha, self.bayesian_test.prior_beta))

    def test_estimate_days_to_run_exp(self):
        """Test estimation of days required for an experiment."""
        estimated_days = self.bayesian_test.estimate_days_to_run_exp(daily_visitors=500, threshold=0.95)
        self.assertTrue(isinstance(estimated_days, int) and estimated_days > 0)

if __name__ == '__main__':
    unittest.main(argv= [''], exit = False)

~~~
