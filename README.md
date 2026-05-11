# Global Inflation Analysis

A data visualization and machine learning dashboard built to analyze post-COVID global inflation trends. The application features a custom, high-contrast minimalist interface and compares predictive models to forecast inflation rates based on global economic factors.

## Overview
This project processes global economic datasets to uncover correlations and build predictive models. The interface prioritizes clean, monochrome data presentation with targeted interactive elements.

## Features
* **Exploratory Data Analysis:** Computes summary statistics (Mean, Median, Mode, Quartiles) for selected countries.
* **Interactive Visualization:** Box plots and scatter matrices to identify outliers and multivariable correlations.
* **Predictive Modeling:** Compares Linear Regression and Random Forest Regressor models to forecast inflation rates using metrics like GDP growth, oil prices, and supply chain indices.

## Project Structure
* `app.py`: The main Streamlit dashboard script.
* `global_inflation_post_covid.csv`: The primary dataset.
* `report.tex` & `Proability & Statistics Project.pdf`: Academic documentation and LaTeX source code.
* `requirements.txt`: Project dependencies.

## Setup Instructions
To run this project locally:

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/global-inflation-analysis.git](https://github.com/yourusername/global-inflation-analysis.git)

2. Install the required dependencies:
  pip install -r requirements.txt

3. Launch the application:
  streamlit run app.py
