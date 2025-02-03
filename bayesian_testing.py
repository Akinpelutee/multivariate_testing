import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


class BayesianMultivariateTest:
    def __init__(self, data_path, prior_alpha=1, prior_beta=1, n_samples=100_000):
        """
        Initialize with the multivariate test data from a CSV file.
        Args:
        - data_path (str): Path to the CSV file containing ['Combination', 'Conversions', 'Visitors'].
        - prior_alpha (int): Alpha parameter for Beta distribution prior.
        - prior_beta (int): Beta parameter for Beta distribution prior.
        - n_samples (int): Number of Monte Carlo samples for probability estimation.
        """
        self.data = pd.read_csv(data_path)
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
        Update the prior for the next day only if the dataset has changed.
        """
        current_data_hash = hash(self.data.to_string())

        if current_data_hash != self.last_data_hash:
            # Only update if data has changed
            for comb, posterior in self.posteriors.items():
                self.prior_alpha = posterior.args[0]  # Alpha from posterior becomes new prior
                self.prior_beta = posterior.args[1]   # Beta from posterior becomes new prior

            self.last_data_hash = current_data_hash  # Update the last known data state
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

    def plot_posteriors(self):
        """Plot the posterior distributions for each combination."""
        x = np.linspace(0, 1, 1000)
        plt.figure(figsize=(10, 6))
        for comb, posterior in self.posteriors.items():
            plt.plot(x, posterior.pdf(x), label=f"{comb} (Beta({posterior.args[0]:.2f}, {posterior.args[1]:.2f}))")
        plt.title("Posterior Distributions")
        plt.xlabel("Conversion Rate")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    def summary(self):
        """Summarize posterior parameters and probabilities of being the best."""
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
        """Estimate number of days required to reach a 95% probability of best variant."""
        np.random.seed(42)
        cumulative_conversions = {comb: 0 for comb in self.data['Combination']}
        cumulative_visitors = {comb: 0 for comb in self.data['Combination']}
        
        for day in range(1, max_days + 1):
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
                return day
        
        return max_days


if __name__ == "__main__":
    test = BayesianMultivariateTest("data.csv")
    test.calculate_posteriors()
    test.plot_posteriors()

    test.update_prior_with_posterior()
    
    summary = test.summary()
    print(summary)
    print("Probabilities of being the best:")
    
    print(test.probability_of_being_best())
    
    days_required = test.estimate_days_to_run_exp(daily_visitors=50)
    print(f"Estimated days to run: {days_required}days")

    CI_mean = test.get_posterior_summary()
    print(CI_mean)