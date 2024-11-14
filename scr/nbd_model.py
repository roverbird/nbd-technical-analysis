import numpy as np
import pandas as pd
import nltk
from statsmodels.discrete import discrete_model
from scr.cramer_von_mises import (
    cramervonmises_test,
)  # Import the custom cvm test function

min_ngram_occurrences = (
    10  # Min occurrence of ngram pattern to be considered for analysis
)


class Model:
    def __call__(
        self, data: pd.DataFrame, datetime_batches: dict, n: int, symbol: str
    ) -> None:
        """
        Main entry point to process the data and create populations for a specific symbol.
        """
        population = self.create_population(data, datetime_batches, n)
        population = self.create_negative_binomial_population(population, symbol)
        return population

    def create_population(
        self, data: pd.DataFrame, datetime_batches: dict, n: int
    ) -> dict:
        """
        Create a population of N-grams from the time series data.
        """
        values = data["price"].values  # Access the 'price' column directly
        population = {}

        # Iterate over datetime batches
        # for epoch, batches in datetime_batches.items():
        for batches in datetime_batches.values():
            for i, datetime_batch in enumerate(batches):
                batch_values = values[data["date"].isin(datetime_batch)]

                # Use the separate method for categorizing price changes
                values_categorical = self.categorize_price_changes(batch_values)

                # Generate N-grams from the categorical values
                for ngram in nltk.ngrams(values_categorical, n=n):
                    if ngram not in population:
                        population[ngram] = np.zeros(len(batches))
                    population[ngram][i] += 1  # Increment the count for the N-gram

        return population

    def create_negative_binomial_population(self, population, symbol: str):
        """
        Create a population of N-grams that fit the Negative Binomial Distribution for a specific symbol.
        """
        negative_binomial_population = {}

        for ngram, values in population.items():
            if (
                sum(values) < min_ngram_occurrences
            ):  # Use the min_ngram_occurrences variable
                continue

            n_estimate, p_estimate = self.mle(values)  # Estimate parameters

            if not n_estimate or not p_estimate:
                continue

            # Perform the Cramer-von Mises test to check distribution fit
            p_value = cramervonmises_test(
                values, n_estimate, p_estimate
            )  # Assume this returns a p-value

            if p_value > 0.05:
                negative_binomial_population[ngram] = (n_estimate, p_estimate)
                # Uncomment for debugging
                #print(
                #    f"Added n-gram: {ngram} with parameters: n_estimate={n_estimate}, p_estimate={p_estimate}, Cramer-von Mises p-value={p_value}"
                #)

        return negative_binomial_population


    @staticmethod
    def categorize_price_changes(batch_values: np.ndarray) -> np.ndarray:
        """
        Categorize the price changes into bins based on the percentage change between consecutive values.
        """
        # Calculate percentage changes between consecutive values
        price_changes_percent = (
            (batch_values[1:] - batch_values[:-1]) / batch_values[:-1] * 100
        )

        # Define bins here # Here you set the way you categorize input variable, i.e. 'price', into n-grams
        categorized_values = pd.cut(
            price_changes_percent,
            bins=[
                -np.inf,  # Very heavy drop
                -2,  # Heavy drop
                -1,  # Moderate drop
                -0.5,  # Light drop
                -0.2,  # Minor drop, near-zero negative
                0.2,  # Unchanged or very minor changes
                0.5,  # Light gain
                1,  # Moderate gain
                2,  # Heavy gain
                np.inf,  # Very heavy gain
            ],
            labels=["VHD", "HD", "MD", "LD", "U", "LG", "MG", "HG", "VHG"],
        )

        return (
            pd.Series(categorized_values).dropna().values
        )  # Drop NA values and return as a numpy array

    @staticmethod
    def mle(values):
        """
        Perform Maximum Likelihood Estimation to find NBD parameters.
        """
        model = discrete_model.NegativeBinomial(
            values, np.ones_like(values)
        )  # Fit the NBD model
        model = model.fit(disp=False)
        mu, alpha = np.exp(model.params[0]), model.params[1]
        p_estimate = 1 / (1 + mu * alpha)
        n_estimate = mu * p_estimate / (1 - p_estimate)
        return n_estimate, p_estimate
