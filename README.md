# Microsoft-Stock-Analysis

## Overview

This repository contains a Python script for analyzing Microsoft's stock data. The script covers various aspects such as data preprocessing, principal component analysis (PCA), ARIMA forecasting, and clustering using the K-Means algorithm.


## Introduction

Understanding the historical performance of a company's stock is crucial for investors and analysts. This Python script provides a comprehensive analysis of Microsoft's stock data, offering insights into trends, patterns, and future predictions.

## Features

1. **Data Loading and Pre-processing:**
   - Load stock data from a CSV file.
   - Clean the data by removing duplicates and handling null values.
   - Split the data into training and test sets.

2. **Principal Component Analysis (PCA):**
   - Apply PCA to reduce dimensionality and identify underlying patterns.

3. **ARIMA Forecasting:**
   - Use the ARIMA model for time series forecasting of stock prices.

4. **Clustering with K-Means:**
   - Find the optimal number of clusters using the Elbow method.
   - Apply K-Means clustering to group similar data points.

5. **User Interaction: Prediction Input:**
   - Allow users to input data for predicting the cluster using the trained K-Means model.
