import numpy as np
from scipy import stats


def cramervonmises_test(sample, n_est, p_est):
    """
    Perform the Cramér-von Mises goodness-of-fit test for a negative binomial distribution.

    Parameters:
        sample (np.array): The sample values to be tested.
        n_est (float): Estimated parameter n of the negative binomial distribution.
        p_est (float): Estimated parameter p of the negative binomial distribution.

    Returns:
        float: The p-value of the test.
    """
    # Generate a large number of random samples from the negative binomial distribution
    theoretical_values = np.random.negative_binomial(n_est, p_est, 1000)

    # Calculate the empirical CDF of the sample
    sample_sorted = np.sort(sample)
    ecdf = np.arange(1, len(sample_sorted) + 1) / len(sample_sorted)

    # Calculate the theoretical CDF for the negative binomial distribution
    theoretical_ecdf = stats.nbinom.cdf(sample_sorted, n_est, p_est)

    # Calculate the Cramér-von Mises statistic
    cvm_statistic = np.sum((ecdf - theoretical_ecdf) ** 2)

    # Estimate the p-value using a bootstrap method (optional step, remove if not needed)
    bootstrap_statistics = []
    for _ in range(1000):
        resampled_data = np.random.choice(
            theoretical_values, size=len(sample), replace=True
        )
        resampled_data_sorted = np.sort(resampled_data)
        resampled_ecdf = stats.nbinom.cdf(resampled_data_sorted, n_est, p_est)
        bootstrap_stat = np.sum(
            (
                np.arange(1, len(resampled_data) + 1) / len(resampled_data)
                - resampled_ecdf
            )
            ** 2
        )
        bootstrap_statistics.append(bootstrap_stat)

    # Calculate the p-value as the proportion of bootstrap statistics greater than the observed statistic
    p_value = np.mean(np.array(bootstrap_statistics) >= cvm_statistic)

    return p_value
