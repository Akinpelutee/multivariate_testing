# Multivariate Testing

### Algorithm walkthrough

This project gives an overview of a Bayesian multivariate test which involves four combinations.
We are to test between four variations in an experiment (Two headers and two button colors). The aim of this experiment is to ascertain a winning variation not based on intuition but by using the bayesian approach to determine the extent to which to which our leading variation is likely to be certain using a uniform prior(Uniform distribution).

### Data

The data used here is just generated manually. It has three features which are:
Combination, Conversions and Visitors.
Combination :- 'H1 + B1' (control group) while 'H1+B2', 'H2+B1', 'H2+B2' are the variations exposed to the experimental group.
Assuming we have 200 users and 50 users are exposed to each variant also having 3,5,7,10 conversions in the first day of running the experiment.

~~~python
# Input Data (Example for day 1)
data_day_1 = pd.DataFrame({
    'Combination': ['H1 + B1', 'H1 + B2', 'H2 + B1', 'H2 + B2'],
    'Conversions': [3, 5, 7, 10],
    'Visitors': [50, 50, 50, 50]
})
~~~

### Functions

#### validate_data()

This function makes sure we have the correct number of features and also handles missing values correctly. It raises a ValueError which prompts user to input any missing data correctly before analysis.

##### calculate_posteriors()

Calculates the posterior Beta distributions for each variant based on prior data and observed conversions.

##### update_prior_with_posterior()

Updates the prior for the next day using the posterior values from the current day. This allows continuous learning from new data.

##### probability_of_being_best(n_samples=100_000)

Uses Monte Carlo simulation to estimate the probability that each variant is the best performer. Instead of relying on a single conversion rate, we simulate many possible scenarios and see which option wins most often.
e.g If H2+B2 has 48% probability of being best, it means that in 48,000 out of the 100,000 simulations, it was the top performer. 

##### plot_posteriors()

Plots the posterior Beta distributions for all tested combinations, showing their estimated conversion rates.

##### summary()

Provides a tabular summary of the experiment, including posterior parameters and probability of each combination being the best.

##### get_posterior_summary()

Returns the mean and 95% credible interval (CI) for each variant’s posterior distribution.

##### estimate_days_to_run_exp(daily_visitors, threshold=0.95, max_days=14)

Simulates the experiment over multiple days, estimating how long it will take for a variant to be confidently declared the best.


### Usage

#### 1) Import the Class
Save bayesian_test.py in your project directory and import it
~~~python
from bayesian_test import BayesianMultivariateTest
import pandas as pd
~~~
#### 2) Update Priors for Continuous Testing
~~~python
test.update_prior_with_posterior()
~~~

#### 3) Run Bayesian Test
~~~python
test = BayesianMultivariateTest(data)
test.calculate_posteriors()
test.plot_posteriors()
~~~

#### 4) Get Probability of Best Variant
~~~python
probabilities = test.probability_of_being_best()
print(probabilities)
~~~

#### 5) Estimate Experiment Duration
~~~python
days_needed = test.estimate_days_to_run_exp(daily_visitors=50, threshold=0.95, max_days=14)
print(f"Estimated days to run: {days_needed}")
~~~


~~~
