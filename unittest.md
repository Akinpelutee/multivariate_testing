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


