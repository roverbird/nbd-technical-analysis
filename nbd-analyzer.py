#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from scr.nbd_model import Model

INPUT_DATA_FOLDER = "../input_data"
RESULTS_FOLDER = "../results_nbd_mean"
NGRAM_SIZE = 1  # Set ngram size, 2 for 2-grams
KLJUV_VALUE = 1  # Filter estimated NBD parameters based on Euclidean distance. The less the value, the more aggregated ("meaningful", or "significant") are the ngrams.
EPOCHS = [3]  # Define epochs for batching (in days)

current_time = pd.Timestamp.now()
END_DATE = current_time - pd.Timedelta(
    minutes=current_time.minute % 60,
    seconds=current_time.second,
    microseconds=current_time.microsecond,
)
START_DATE = END_DATE - pd.Timedelta(days=303)

print(f"==============================================")
print(f"Output dir: {RESULTS_FOLDER}...")
print(f"Input dir: {INPUT_DATA_FOLDER}")
print(f"Start Date: {START_DATE}")
print(f"End Date: {END_DATE}")
print(f"Execution time: {current_time}")
print(f"==============================================")


def read_time_series_data(folder):
    """
    Reads CSV files from the specified folder and processes them into a single DataFrame.
    Files are processed in alphabetical order.
    """
    print("Reading CSV files from the project folder...")

    # Get a sorted list of CSV filenames
    filenames = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])

    # Load all files at once into a list of DataFrames
    # Define structure of input data csv here
    dataframes = [
        pd.read_csv(
            os.path.join(folder, filename),
            parse_dates=["future_timestamp"],
            usecols=[
                "future_timestamp",
                "symbol",
                "price",
                "30_day_hourly_future_return",
            ],
        ).rename(
            columns={
                "future_timestamp": "date", # set 'date' column here
                "30_day_hourly_future_return": "open",  # set the target column here
                "price": "price", # generate features from this column
            }
        )
        for filename in filenames
    ]

    # Concatenate all DataFrames
    full_series = pd.concat(dataframes, ignore_index=True)
    full_series = full_series[full_series["date"] >= START_DATE]
    print(
        f"Loaded {len(filenames)} files into one DataFrame of shape: {full_series.shape}"
    )

    # Pivot the DataFrame to have a dictionary of symbols
    time_series = {symbol: data for symbol, data in full_series.groupby("symbol")}

    print("Time series data successfully loaded.")
    return time_series


def create_batches(dates):
    """Groups dates into batches based on the defined epochs."""
    datetime_batches = {epoch: [] for epoch in EPOCHS}

    for epoch in EPOCHS:
        batch_start = dates[0]
        while batch_start < dates[-1]:
            batch_end = batch_start + pd.Timedelta(days=epoch)
            datetime_batches[epoch].append(
                dates[(dates >= batch_start) & (dates < batch_end)]
            )
            batch_start = batch_end

    return datetime_batches


def process_time_series(time_series, datetime_batches, model):
    """Processes each time series to estimate parameters and store them."""
    parameters = {}
    print("Processing each time series to estimate parameters...")

    for series_name, series in time_series.items():
        data = series.copy()  # Keep 'open' as is
        # Pass the series_name as the symbol to the model
        output = model(data, datetime_batches, n=NGRAM_SIZE, symbol=series_name)
        parameters[series_name] = output
        print(f"Processed series: {series_name} with {len(output)} n-grams.")

    return parameters


# def load_valid_ngrams(file_path):
#    """Loads valid N-grams from a temporary file."""
#    if os.path.exists(file_path):
#        valid_ngrams_df = pd.read_csv(file_path)
#        return set(tuple(ngram) for ngram in valid_ngrams_df[['ngram', 'n_estimate', 'p_estimate']].values)
#    return set()
#


def filter_parameters(parameters, kljuv_value):
    """Filters parameters based on Euclidean distance condition using vectorized operations."""
    kljuv_threshold = kljuv_value**2  # Use squared value to avoid the sqrt operation
    return {
        k: {k_: v_ for k_, v_ in v.items() if np.sum(np.square(v_)) < kljuv_threshold}
        for k, v in parameters.items()
    }


def collect_results(parameters, model, time_series, datetime_batches):
    """Collects results for each time series based on estimated parameters."""
    result = {}
    patterns = set(
        (~pd.DataFrame.from_dict(parameters).isna())
        .sum(axis=1)
        .sort_values()
        .tail(5)
        .index
    )

    print("Collecting results for each time series...")

    for name, values in parameters.items():
        result[name] = {}

        # Debug: Notify which time series is being processed
        print(f"Processing symbol: {name}")

        # Create population for this symbol and store results per epoch
        tmp = model.create_population(time_series[name], datetime_batches, n=NGRAM_SIZE)

        # for epoch_index, epoch_batches in enumerate(datetime_batches[EPOCHS[0]]):
        for epoch_index in range(len(datetime_batches[EPOCHS[0]])):
            result[name][epoch_index] = {}

            # Calculate total occurrences in the current epoch to normalize the mean
            total_occurrences_in_epoch = sum(
                tmp[pattern][epoch_index].sum()
                for pattern in patterns
                if pattern in tmp
            )

            for pattern in patterns:
                if pattern in tmp:
                    # Normalize the mean by dividing the counts by the total occurrences in this epoch
                    mean_value = (
                        tmp[pattern][epoch_index].sum() / total_occurrences_in_epoch
                        if total_occurrences_in_epoch > 0
                        else 0
                    )

                    # Print the normalized mean value for debugging
                    # print(f"Symbol: {name}, Epoch: {epoch_index}, Pattern: {pattern}, Normalized Mean Value: {mean_value}")

                    result[name][epoch_index][pattern] = mean_value
                else:
                    result[name][epoch_index][pattern] = 0

                    # Print when a pattern is not found in tmp
                    # print(f"Symbol: {name}, Epoch: {epoch_index}, Pattern: {pattern} not found. Normalized Mean Value: 0")

    return result, patterns


def save_mean_values(result, datetime_batches, time_series):
    """Saves mean values to separate CSV files for each symbol in the RESULTS_FOLDER."""

    # Extract epoch end dates from datetime_batches
    epoch_end_dates = [
        batch[-1] for batches in datetime_batches.values() for batch in batches
    ]

    for symbol, counts in result.items():
        df = pd.DataFrame(index=epoch_end_dates)
        df.index.name = "epoch_end_date"

        symbol_series = time_series[symbol].set_index("date")

        # Add 'symbol' column with the symbol value for all rows
        df["symbol"] = symbol

        # Map the 'last_open' values from time_series to the new DataFrame
        df["last_open"] = symbol_series.reindex(df.index)["open"]

        # Populate the DataFrame with normalized mean n-gram values
        for epoch_index, ngram_counts in counts.items():
            for ngram, mean_value in ngram_counts.items():
                # Convert ngram tuple to string for column name
                ngram_str = str(
                    ngram
                )  # or join with a delimiter, e.g., '_'.join(ngram)

                if ngram_str not in df.columns:
                    df[ngram_str] = 0.0  # Ensure it is float

                # Assign the mean value, explicitly casting to float
                df.at[epoch_end_dates[epoch_index], ngram_str] = float(
                    mean_value
                )  # Ensure it's float

        # Define output file path for mean values
        output_file = os.path.join(RESULTS_FOLDER, f"{symbol}_mean_nbd_results.csv")

        # Save DataFrame to CSV
        df.to_csv(output_file, index=True)
        # print(f"Mean values saved to {output_file}")


def save_results(result, patterns):
    """Saves overall results to a single CSV file with symbols as the first column and n-gram patterns as subsequent columns in the RESULTS_FOLDER."""
    unique_patterns = sorted(set(patterns))  # Get unique n-gram patterns
    all_data = []  # To store all rows of the DataFrame

    # Loop through the result dictionary to prepare rows for each symbol
    for symbol, counts_per_epoch in result.items():
        # Initialize a dictionary for each symbol, with all n-grams set to 0
        symbol_row = {pattern: 0 for pattern in unique_patterns}
        symbol_row["symbol"] = symbol  # Add symbol as the first column

        # For each epoch, sum up occurrences for this symbol
        for epoch_counts in counts_per_epoch.values():
            for pattern, occurrence in epoch_counts.items():
                if pattern in symbol_row:
                    symbol_row[pattern] += occurrence

        all_data.append(symbol_row)  # Append the symbol row data

    # Convert all collected data into a DataFrame
    result_df = pd.DataFrame(all_data)

    # Reorder columns to make sure 'symbol' is the first column
    columns_order = ["symbol"] + unique_patterns
    result_df = result_df[columns_order]

    # Save the DataFrame to a CSV file in RESULTS_FOLDER
    output_file = os.path.join(RESULTS_FOLDER, "nbd_results.csv")
    result_df.to_csv(output_file, index=False)  # Save without the index
    print(f"Results saved to {output_file}")


def main():
    # Load time series data
    time_series = read_time_series_data(INPUT_DATA_FOLDER)

    # Concatenate all time series into a single DataFrame
    time_series_concat = pd.concat(time_series.values(), keys=time_series.keys())
    print(f"Concatenated time series DataFrame shape: {time_series_concat.shape}")

    # Set the date range for analysis
    dates = pd.date_range(START_DATE, END_DATE, freq="h").intersection(
        time_series_concat["date"]
    )  # Hourly frequency

    # Create batches
    datetime_batches = create_batches(dates)

    # Store parameters for each time series
    model = Model()
    parameters = process_time_series(time_series, datetime_batches, model)

    # Load valid N-grams
    # valid_ngrams = load_valid_ngrams("valid_ngrams.csv")

    # Filter parameters
    filtered_parameters = filter_parameters(parameters, KLJUV_VALUE)

    # Collect results
    result, patterns = collect_results(
        filtered_parameters, model, time_series, datetime_batches
    )

    # Save overall results
    save_results(result, patterns)

    # Save mean values for each symbol
    save_mean_values(result, datetime_batches, time_series)


if __name__ == "__main__":
    main()
