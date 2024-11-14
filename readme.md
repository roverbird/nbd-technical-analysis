# N-Gram Trading Data Analysis with Negative Binomial Distribution

This project provides python scripts for modeling trading data using the Negative Binomial Distribution (NBD). Specifically, it performs n-gram analysis on trading data to measure significant n-gram patterns. The NBD parameters are extracted, analyzed, and significant n-grams are filtered to understand patterns in price data, with the end goal of calculating the mean occurrence of these patterns for each asset. The aim of the program is to automatically extract and generate interesting features that are based on price dynamics within trading sessions during designated period (called 'day batches', size of batch is set in EPOCHS var).

Программа для автоматического анализа данных торговых сессий с использованием отрицательного биномиального распределения (NBD): поиск значимых категориальных рядов (н-грамм) заданного размера на основе анализа торгов.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Example Data](#example-data)
- [Function Descriptions](#function-descriptions)
- [Configuration](#configuration)
- [License](#license)

## Project Overview

The code:
1. **Categorizes price data into n-grams**: N-grams of the specified size are generated from the price data.
2. **Applies the Negative Binomial Distribution**: Each n-gram is tested for goodness-of-fit to the NBD, and parameters are extracted.
3. **Filters significant n-grams**: Only n-grams with specific NBD parameters are retained based on a configurable significance filter.
4. **Calculates mean occurrences**: For each asset (one input file per asset), the code computes the mean occurrence of significant n-grams, returning these as outputs.

## Features

- **Dynamic Data Loading**: Reads all CSV files in the specified folder.
- **Flexible Epoch Batching**: Allows configurable batching for temporal analysis.
- **N-Gram Analysis**: Customizable n-gram size to capture various data aggregation levels.
- **Negative Binomial Distribution Fitting**: Fits n-grams to the NBD and extracts parameters for statistical analysis.
- **Significance Filtering**: Configurable filtering to retain only significant n-grams based on a Euclidean distance threshold.
- **Result Output**: Generates CSVs summarizing mean occurrences of significant n-grams features for each asset as a time series 
- **nbd_results.csv**: Mean ngram occurrence features for for all assets together in one file (so that you can profile or cluster assets).

## Folder Structure

- **`scr/nbd_model.py`**: Module with the `Model` class implementing NBD fitting logic.
- **`/input_data`**: Folder containing input CSV files, one per asset.
- **`/results_nbd_mean`**: Folder for storing results in CSV format.

## Setup

### Requirements

- Python 3.x
- Pandas, NumPy
- Ensure the input data files are in the `results_prepared` directory.

Install the required packages:

```bash
pip install numpy pandas
```

### Configurations

In the main script, set paths for input and output folders, and configure:
- **NGRAM_SIZE**: Set the n-gram size for data aggregation (e.g., 1 for 1-grams, 2 for 2-grams).
- **KLJUV_VALUE**: NBD params Euclidean distance threshold to filter aggregated n-grams.
- **EPOCHS**: Number of days for batching data (you can start from 1 day)

### Usage

Run the script as follows:

```bash
python main.py
```

### Example Data

Each input CSV file should contain the following columns:

| future_timestamp | symbol | price | 30_day_hourly_future_return |
|------------------|--------|-------|-----------------------------|
| 2021-01-01 00:00 | ABC    | 100.5 | 0.002                      |

- **future_timestamp**: Timestamp for the data point.
- **symbol**: Asset identifier.
- **price**: Price at the given timestamp.
- **30_day_hourly_future_return**: Future return (used as a target variable in NBD fitting).

You can modify headers in the respective part of the main program. 

## Function Descriptions

### `read_time_series_data(folder)`
Reads and combines CSV files from a given folder into a single DataFrame, processing each file for the relevant columns.

### `create_batches(dates)`
Divides date ranges into epochs as defined in the configuration, batching data for analysis.

### `process_time_series(time_series, datetime_batches, model)`
Processes time series data to estimate NBD parameters for each n-gram in each asset's data.

### `filter_parameters(parameters, kljuv_value)`
Filters the NBD parameters to retain only those meeting the Euclidean distance threshold.

### `collect_results(parameters, model, time_series, datetime_batches)`
Collects and aggregates results for each asset by computing the mean occurrences of significant n-grams.

### `save_results(result, patterns)`
Saves a summary of the results to a CSV file with each asset's n-gram pattern occurrences.

### `save_mean_values(result, datetime_batches, time_series)`
Saves detailed results with the mean values of each significant n-gram for each asset.

## Configuration

Modify constants at the beginning of the script to customize:
- **Data paths**: Adjust `INPUT_DATA_FOLDER` and `RESULTS_FOLDER`.
- **Time frame**: Define `START_DATE` and `END_DATE`.
- **Analysis settings**: Configure `NGRAM_SIZE`, `KLJUV_VALUE`, and `EPOCHS`.

## License

This project is licensed under the MIT License.
