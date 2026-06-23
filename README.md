# Customer Segmentation using Machine Learning

## Overview

Customer Segmentation is a machine learning project that groups customers based on their demographic information and spending behavior. The project uses the K-Means clustering algorithm to identify distinct customer segments, helping businesses understand customer preferences and make data-driven marketing decisions.

## Features

* Data preprocessing and analysis
* Customer behavior visualization
* Feature scaling using StandardScaler
* Optimal cluster identification using the Elbow Method
* Customer segmentation using K-Means Clustering
* Cluster distribution analysis
* Export segmented customer data to CSV
* Interactive visualizations using Matplotlib and Seaborn

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Dataset

The dataset contains the following attributes:

| Column Name   | Description                                          |
| ------------- | ---------------------------------------------------- |
| CustomerID    | Unique customer identifier                           |
| Age           | Customer age                                         |
| AnnualIncome  | Annual income of the customer                        |
| SpendingScore | Spending score assigned based on purchasing behavior |

## Project Workflow

1. Load and analyze customer data.
2. Perform data cleaning and preprocessing.
3. Visualize customer demographics and spending patterns.
4. Scale features using StandardScaler.
5. Apply the Elbow Method to determine the optimal number of clusters.
6. Train the K-Means clustering model.
7. Assign customers to clusters.
8. Visualize customer segments and cluster distribution.
9. Save the segmented customer dataset.

## Installation

Clone the repository:

bash
git clone https://github.com/your-username/customer-segmentation.git

Navigate to the project directory:

bash
cd customer-segmentation


Install required dependencies:

bash
pip install pandas numpy matplotlib seaborn scikit-learn

## Usage

Run the project:

bash
python customer_segmentation.py

## Output

The project generates:
* Age Distribution Visualization
* Income vs Spending Score Analysis
* Elbow Method Graph
* Customer Segmentation Scatter Plot
* Cluster Distribution Chart
* Segmented Customer Dataset (CSV)

## Learning Outcomes

* Customer Analytics
* Unsupervised Machine Learning
* K-Means Clustering
* Data Visualization
* Feature Engineering
* Business Intelligence and Customer Insights

## Author
Arun A
Artificial Intelligence and Data Science Student
