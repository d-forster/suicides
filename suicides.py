'''
This file contains all of the code dealing with data analysis + representation.

TODO 
'''

# Package imports
import numpy as np
import scipy as scp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading dataset
train_file_path = "./input/master.csv"
dataset_df = pd.read_csv(train_file_path)
print("Loading dataset CSV...")
print("Full dataset shape is {}\n".format(dataset_df.shape))
print("Here's the first few rows:\n")
print(dataset_df.head(3))
print()

unique_countries = len(dataset_df["country"].unique())
unique_ages = len(dataset_df["age"].unique())
first_year = min(dataset_df["year"])
last_year = max(dataset_df["year"])
print("There are {} countries listed, with {} different age categories.\n".format(unique_countries, unique_ages))
print("The data set covers the years {} to {}.\n".format(first_year, last_year))

# Cleaning up column headings:
dataset_df = dataset_df.rename(columns={'suicides/100k pop': 'suicides/100k',
                                        'HDI for year': 'HDI_for_year',
                                        ' gdp_for_year ($) ': 'gdp_for_year', 
                                        'gdp_per_capita ($)': 'gdp_per_capita'})
print("The column headings are: {}".format(dataset_df.columns))

# Visualisation of raw data
# 
# First, amount of data by country:
plt.figure(figsize=(10,20))
sns.countplot(y='country', data=dataset_df, alpha=0.7)
plt.title("Amount of data by country")
plt.ylabel("Country")
plt.xlabel("Amount of data points")
plt.tight_layout()
plt.savefig("./output/data_by_country.png")

# Now GDP by country
plt.figure(figsize=(20, 15))
# country_gdp_years = dataset_df[["country-year", 9]].drop_duplicates()
# print(country_years)