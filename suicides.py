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
import warnings

warnings.filterwarnings("ignore")

# Reading dataset
train_file_path = "./input/master.csv"
dataset_df = pd.read_csv(train_file_path)
# print("Loading dataset CSV...")
# print("Full dataset shape is {}\n".format(dataset_df.shape))
# print("Here's the first few rows:\n")
# print(dataset_df.head(3))
# print()

unique_countries = len(dataset_df["country"].unique())
unique_ages = len(dataset_df["age"].unique())
first_year = min(dataset_df["year"])
last_year = max(dataset_df["year"])
# print("There are {} countries listed, with {} different age categories.\n".format(unique_countries, unique_ages))
# print("The data set covers the years {} to {}.\n".format(first_year, last_year))

# Cleaning up data:
dataset_df = dataset_df.rename(columns={'suicides/100k pop': 'suicides/100k',
                                        'HDI for year': 'HDI_for_year',
                                        ' gdp_for_year ($) ': 'gdp_for_year', 
                                        'gdp_per_capita ($)': 'gdp_per_capita'})
# print("The column headings are: {}".format(dataset_df.columns))

dataset_df['gdp_for_year'] = dataset_df['gdp_for_year'].replace(',','', regex=True)
dataset_df['gdp_for_year'] = pd.to_numeric(dataset_df['gdp_for_year'])

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
print("Data by country plot complete.")

# Now GDP by country

def get_cmap(n, name='hsv'):
    """Returns a function that maps each index 0, 1, ..., n-1 to a distinct RGB color. The keyword argument "name" must be a standard mp1 colormap name.

    Args:
        n (int): The number of indices to be used.
        name (str, optional): The name od the mp1 colormap to be used. Defaults to 'hsv'.
    """
    return plt.cm.get_cmap(name, n)

def improve_legend(ax=None):
    """Adds labels to the end of each line according to the legend.

    Args:
        ax (matplotlib axes, optional): The axes of the plot to be improved. Defaults to None.
    """    
    if ax is None:
        ax = plt.gca()

    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    
    for line in ax.lines:
        data_x, data_y = line.get_data()
        rightmost_x = data_x[-1]
        rightmost_y = data_y[-1]
        ax.annotate(
            line.get_label(),
            xy = (rightmost_x, rightmost_y),
            xytext = (5, 0),
            textcoords = "offset points",
            va = "center",
            color = line.get_color()
        )
    ax.legend().set_visible(False)

country_gdp_years = dataset_df[["country", "year", "gdp_for_year"]].drop_duplicates()
# print(country_gdp_years)
plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(40,40))
plt.ylim([min(country_gdp_years['gdp_for_year']), max(country_gdp_years['gdp_for_year'])])
plt.xlim([min(country_gdp_years['year']), max(country_gdp_years['year'])])
colormap = get_cmap(unique_countries)
for i, country in enumerate(country_gdp_years['country'].drop_duplicates()):
    select_country_data = country_gdp_years[country_gdp_years['country'] == country]
    plt.plot(select_country_data['year'], select_country_data['gdp_for_year'], label=str(country), c=colormap(i), linewidth=2)
# print(last_years)
plt.legend()
improve_legend()
plt.xlabel("Year")
plt.ylabel("GDP ($10^{13}$ USD)")
plt.title("GDP for each country over time")
plt.tight_layout()
plt.savefig("./output/GDP_by_country.png")
print("GDP by country plot complete.")

#Redo of the above, but remove the top few countries to see the bottom end more clearly.
trimmed_country_gdp_years = country_gdp_years[(country_gdp_years['country'] != 'United States') & 
                                              (country_gdp_years['country'] != 'France') &
                                              (country_gdp_years['country'] != 'United Kingdom') &
                                              (country_gdp_years['country'] != 'Germany') &
                                              (country_gdp_years['country'] != 'Japan')]
# print("Is USA present in trimmed GDP data?: --> {}".format('United States' in trimmed_country_gdp_years['country']))

plt.figure(figsize=(40,40))
plt.ylim([min(trimmed_country_gdp_years['gdp_for_year']), max(trimmed_country_gdp_years['gdp_for_year'])])
plt.xlim([min(trimmed_country_gdp_years['year']), max(trimmed_country_gdp_years['year'])])
colormap = get_cmap(len(trimmed_country_gdp_years['country'].drop_duplicates()))
for i, country in enumerate(trimmed_country_gdp_years['country'].drop_duplicates()):
    trimmed_select_country_data = trimmed_country_gdp_years[trimmed_country_gdp_years['country'] == country]
    plt.plot(trimmed_select_country_data['year'], trimmed_select_country_data['gdp_for_year'], label=str(country), c=colormap(i), linewidth=2)
plt.legend()
improve_legend()
plt.xlabel("Year")
plt.ylabel("GDP ($10^{12}$ USD)")
plt.title("GDP for trimmed list of countries over time")
plt.tight_layout()
plt.savefig("./output/GDP_by_country_trimmed.png")
print("Trimmed GDP by country plot complete.")