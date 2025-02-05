import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import base64
from scipy.stats import beta

class BayesianMultivariateTest:
    def __init__(self, csv_content, prior_alpha=1, prior_beta=1, n_samples=100_000):
        """
        Initialize with the multivariate test data from a Base64-decoded CSV content.
        Args:
        - csv_content (str): Base64-decoded CSV content.
        - prior_alpha (int): Alpha parameter for Beta distribution prior.
        - prior_beta (int): Beta parameter for Beta distribution prior.
        - n_samples (int): Number of Monte Carlo samples for probability estimation.
        """
        from io import StringIO
        decoded_csv = base64.b64decode(csv_content).decode("utf-8")
        self.data = pd.read_csv(StringIO(decoded_csv))
        
        self.validate_data()
        self.data['NonConversions'] = self.data['Visitors'] - self.data['Conversions']
        
        self.posteriors = {}
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.n_samples = n_samples
        self.last_data_hash = hash(self.data.to_string())  # Track changes in data

    def validate_data(self):
        """Validate if required columns exist and handle missing values."""
        required_columns = ["Combination", "Conversions", "Visitors"]
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        if self.data.isnull().any().any():
            raise ValueError("Dataset contains missing values. Please clean the data before running the analysis.")

    def calculate_posteriors(self):
        """Calculate posterior distributions for each combination."""
        self.posteriors = {
            row["Combination"]: beta(
                self.prior_alpha + row["Conversions"],
                self.prior_beta + (row["Visitors"] - row["Conversions"])
            )
            for _, row in self.data.iterrows()
        }
        return self.posteriors

    def update_prior_with_posterior(self):
        """
        Update the prior for the next day only if the dataset has changed.
        """
        current_data_hash = hash(self.data.to_string())

        if current_data_hash != self.last_data_hash:
            for comb, posterior in self.posteriors.items():
                self.prior_alpha = posterior.args[0]  # Update alpha
                self.prior_beta = posterior.args[1]   # Update beta

            self.last_data_hash = current_data_hash
            print("Priors updated based on new data.")
        else:
            print("No new data detected. Priors remain unchanged.")

    def probability_of_being_best(self):
        """Estimate probability that each combination is the best."""
        sampled_rates = {comb: posterior.rvs(self.n_samples) for comb, posterior in self.posteriors.items()}
        sampled_df = pd.DataFrame(sampled_rates)
        best_combination = sampled_df.idxmax(axis=1)
        probabilities = best_combination.value_counts(normalize=True)
        return probabilities.reindex(self.data['Combination'].unique(), fill_value=0)

    def estimate_days_to_run_exp(self, daily_visitors, threshold=0.95):
        """Estimate number of days required to reach a probability threshold for the best variant."""
        np.random.seed(42)
        cumulative_conversions = {comb: 0 for comb in self.data['Combination']}
        cumulative_visitors = {comb: 0 for comb in self.data['Combination']}
        
        days = 0
        while True:  # No max_days limit
            days += 1
            for _, row in self.data.iterrows():
                comb = row['Combination']
                true_rate = row['Conversions'] / row['Visitors']
                daily_conversions = np.random.binomial(daily_visitors, true_rate)
                cumulative_conversions[comb] += daily_conversions
                cumulative_visitors[comb] += daily_visitors

            self.posteriors = {
                comb: beta(1 + cumulative_conversions[comb], 1 + (cumulative_visitors[comb] - cumulative_conversions[comb]))
                for comb in cumulative_conversions
            }
            
            probabilities = self.probability_of_being_best()
            if probabilities.max() >= threshold:
                return days

    def get_posterior_summary(self):
        """Get the mean and 95% credible interval (CI) for each combination."""
        CI_mean = {}
        for comb, posterior in self.posteriors.items():
            mean = posterior.mean()
            lower, upper = posterior.interval(0.95)
            CI_mean[comb] = {
                'mean': mean,
                '95% CI lower': lower,
                '95% CI upper': upper
            }
        return CI_mean

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 bayesian_testing.py BASE64_ENCODED_CSV_CONTENT")
        sys.exit(1)

    base64_csv_content = sys.argv[1]
    
    test = BayesianMultivariateTest(base64_csv_content)
    test.calculate_posteriors()
    
    test.update_prior_with_posterior() 
    
    print("Posterior Summary:", test.get_posterior_summary())

    print("Probabilities of being the best:", test.probability_of_being_best())

    days_required = test.estimate_days_to_run_exp(daily_visitors=50)
    print(f"Estimated days to run: {days_required} days")



import numpy as np
import argparse
from scipy.stats import beta

def bayesian_duration(
    baseline_rate,  
    mde,            
    daily_traffic,  
    num_variations=4,  
    threshold=0.99,  
    max_days=1000      
):
    """Calculate Bayesian test duration based on given parameters."""
    np.random.seed(42)
    
    # Initialize priors with a stronger belief to speed up convergence
    priors = {f'variation_{i}': {'alpha': 10, 'beta': 30} for i in range(num_variations)}
    
    # Use relative MDE
    true_rates = [baseline_rate] + [baseline_rate * (1 + mde)] * (num_variations - 1)
    
    traffic_per_variation = daily_traffic // num_variations
    control_key = 'variation_0'
    
    for day in range(1, max_days + 1):
        daily_results = {}
        
        # Simulate conversions
        for i, var in enumerate(priors.keys()):
            conversions = np.random.binomial(traffic_per_variation, true_rates[i])
            daily_results[var] = conversions
        
        # Update posteriors
        for var in priors.keys():
            priors[var]['alpha'] += daily_results[var]
            priors[var]['beta'] += (traffic_per_variation - daily_results[var])
        
        # Thompson Sampling with a large sample size for stability
        samples = {var: beta.rvs(priors[var]['alpha'], priors[var]['beta'], size=100000) 
                  for var in priors.keys()}
        
        # Compare variations against control
        control_samples = samples[control_key]
        best_prob = max(np.mean(samples[var] > control_samples) for var in samples if var != control_key)

        # Stop if a variation is significantly better than control
        if best_prob >= threshold:
            return day
    
    return max_days


if __name__ == "__main__":
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Calculate Bayesian test duration.")
    
    parser.add_argument("--baseline", type=float, required=True, help="Baseline conversion rate (e.g., 0.1 for 10%)")
    parser.add_argument("--mde", type=float, required=True, help="Minimum detectable effect (e.g., 0.1 for 10%)")
    parser.add_argument("--daily_visitors", type=int, required=True, help="Total daily traffic")
    parser.add_argument("--variations", type=int, default=4, help="Number of variations (including control)")
    
    args = parser.parse_args()

    days_needed = bayesian_duration(
        baseline_rate=args.baseline,
        mde=args.mde,
        daily_traffic=args.daily_visitors,
        num_variations=args.variations
    )

    print(f"Estimated days required to run experiment: {days_needed} days")

