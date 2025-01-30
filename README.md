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

##### calculate_posteriors()

Calculates the posterior Beta distributions for each variant based on prior data and observed conversions.

##### update_prior_with_posterior()

Updates the prior for the next day using the posterior values from the current day. This allows continuous learning from new data.

##### probability_of_being_best(n_samples=100_000)

Uses Monte Carlo simulation to estimate the probability that each variant is the best performer.

##### plot_posteriors()

Plots the posterior Beta distributions for all tested combinations, showing their estimated conversion rates.

##### summary()

Provides a tabular summary of the experiment, including posterior parameters and probability of each combination being the best.

##### get_posterior_summary()

Returns the mean and 95% credible interval (CI) for each variant’s posterior distribution.

##### estimate_days_to_run_exp(daily_visitors, threshold=0.95, max_days=14)

Simulates the experiment over multiple days, estimating how long it will take for a variant to be confidently declared the best.


~~~python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

class BayesianMultivariateTest:
    def __init__(self, data):
        """
        Initialize with the multivariate test data.
        Args:
        - data (DataFrame): A DataFrame with columns ['Combination', 'Conversions', 'Visitors'].
        """
        self.data = data
        self.data['NonConversions'] = self.data['Visitors'] - self.data['Conversions']
        self.data['Alpha'] = 1 + self.data['Conversions']
        self.data['Beta'] = 1 + self.data['NonConversions']
        self.posteriors = {}
        
        # Initialize priors for the first day (can be adjusted if needed)
        self.prior_alpha = 1
        self.prior_beta = 1

    def calculate_posteriors(self):
        """Calculate posterior distributions for each combination."""
        # Ensure required columns exist
        required_columns = ["Combination", "Conversions", "Visitors"]
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate posterior distributions
        self.posteriors = {
            row["Combination"]: beta(
                self.prior_alpha + row["Conversions"],
                self.prior_beta + (row["Visitors"] - row["Conversions"])
            )
            for _, row in self.data.iterrows()
        }
        
        # Debugging: Print or log the posteriors
        for Combination, posterior in self.posteriors.items():
            print(f"Combination: {Combination}, Posterior: Beta({posterior.args[0]:.2f}, {posterior.args[1]:.2f})")
        
        # Optionally return the posteriors
        return self.posteriors

    def update_prior_with_posterior(self):
        """
        Update the prior for the next day based on the posterior of the current day.
        """
        # Calculate posterior for each combination and update prior
        for Combination, posterior in self.posteriors.items():
            # Update the prior for the next day
            self.prior_alpha = posterior.args[0]  # Alpha from the posterior becomes the prior for the next day
            self.prior_beta = posterior.args[1]   # Beta from the posterior becomes the prior for the next day
        
    def probability_of_being_best(self, n_samples=100_000):
        """
        Calculate the probability of each combination being the best.
        Args:
        - n_samples (int): Number of samples for Monte Carlo simulation.
        """
        # Sample conversion rates from posterior distributions
        sampled_rates = {
            Combination: posterior.rvs(n_samples)
            for Combination, posterior in self.posteriors.items()
        }
        
        # Create a DataFrame to hold sampled rates
        sampled_rates_df = pd.DataFrame(sampled_rates)

        # Find the combination with the highest rate for each sample
        best_combination = sampled_rates_df.idxmax(axis=1)

        # Calculate the probability of each combination being the best
        probabilities = best_combination.value_counts(normalize=True)
        
        # Fill in probabilities for all combinations (in case some have 0 probability)
        all_combinations = self.data['Combination'].unique()
        probabilities = probabilities.reindex(all_combinations, fill_value=0)
        
        return probabilities

    def plot_posteriors(self):
        """
        Plot the posterior distributions for each combination.
        """
        x = np.linspace(0, 1, 1000)
        plt.figure(figsize=(10, 6))

        for Combination, posterior in self.posteriors.items():
            y = posterior.pdf(x)
            plt.plot(x, y, label=f"{Combination} (Beta({posterior.args[0]}, {posterior.args[1]}))")

        plt.title("Posterior Distributions of Combinations")
        plt.xlabel("Conversion Rate")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()
        plt.show()

    def summary(self):
        """
        Summarize the posterior parameters and probabilities of being the best.
        """
        self.calculate_posteriors()
        probabilities = self.probability_of_being_best()
        summary = self.data.copy()
        summary['Probability of Being Best'] = summary['Combination'].map(probabilities)
        return summary

    def get_posterior_summary(self):

        """
        Gets the mean of the posterior distribution with a credibility interval at 95%
        """
        # Get the mean and 95% credible interval (CI) for each combination
        CI_mean = {}
        for comb, posterior in self.posteriors.items():
            mean = posterior.mean()
            lower, upper = posterior.interval(0.95)  # 95% CI
            CI_mean[comb] = {
                'mean': mean,
                '95% CI lower': lower,
                '95% CI upper': upper
            }
        return CI_mean

    def estimate_days_to_run_exp(self, daily_visitors, threshold=0.95, max_days=14):
        """
        Estimate the number of days required to run the experiment.
        
        Args:
        - daily_visitors (int): Number of visitors per day per combination.
        - threshold (float): Probability threshold for one combination being the best.
        - max_days (int): Maximum number of days to simulate.
        
        Returns:
        - days (int): Estimated number of days required to meet the threshold.
        """
        # Initialize running totals for each combination
        np.random.seed(42)
        cumulative_conversions = {comb: 0 for comb in self.data['Combination']}
        cumulative_visitors = {comb: 0 for comb in self.data['Combination']}
        
        for day in range(1, max_days + 1):
            # Simulate daily visitors for each combination
            for _, row in self.data.iterrows():
                comb = row['Combination']
                true_rate = row['Conversions'] / row['Visitors']  # True conversion rate (assumed)
                daily_conversions = np.random.binomial(daily_visitors, true_rate)
                
                # Update cumulative data
                cumulative_conversions[comb] += daily_conversions
                cumulative_visitors[comb] += daily_visitors
            
            # Update posteriors
            self.posteriors = {
                comb: beta(
                    1 + cumulative_conversions[comb],
                    1 + (cumulative_visitors[comb] - cumulative_conversions[comb])
                )
                for comb in cumulative_conversions
            }
            
            # Update prior for the next day based on the current day's posterior
            self.update_prior_with_posterior()
            
            # Calculate probabilities of being the best
            probabilities = self.probability_of_being_best()
            
            # Check if any combination exceeds the threshold
            if probabilities.max() >= threshold:
                return day
        
        # If max_days is reached without convergence
        return max_days


# Input Data (Example for day 1)
data = pd.DataFrame({
    'Combination': ['H1 + B1', 'H1 + B2', 'H2 + B1', 'H2 + B2'],
    'Conversions': [3, 5, 7, 10],
    'Visitors': [50, 50, 50, 50]
})


# Run Bayesian Multivariate Test
test = BayesianMultivariateTest(data_day_1)
posteriors_day_1 = test.calculate_posteriors()

#test.update_prior_with_posterior()

# Plot Posterior Distributions
test.plot_posteriors()

# Show Summary
summary = test.summary()
print(summary)

# Show Probabilities of Being the Best
probabilities = test.probability_of_being_best()
print("\nProbabilities of Each Combination Being the Best:")
print(probabilities)

# Estimate the number of days required to run the experiment
daily_visitors = 50  # Assume 50 visitors per day per combination
days_required = test.estimate_days_to_run_exp(daily_visitors, threshold=0.95)

print(f"Estimated days required to run the experiment: {days_required} days\n")

CI_mean = test.get_posterior_summary()
print(CI_mean)

~~~

### Usage

#### 1) Import the Class
Save bayesian_test.py in your project directory and import it
~~~python
from bayesian_test import BayesianMultivariateTest
import pandas as pd
~~~

#### 2) Load Data
~~~python
data = pd.DataFrame({
    'Combination': ['H1 + B1', 'H1 + B2', 'H2 + B1', 'H2 + B2'],
    'Conversions': [3, 5, 7, 10],
    'Visitors': [50, 50, 50, 50]
})
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
#### 6) Update Priors for Continuous Testing
~~~python
test.update_prior_with_posterior()
~~~

~~~
