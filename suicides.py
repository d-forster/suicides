'''
This file contains all of the code dealing with data analysis + representation.

TODO 
'''


"""
********************************************
PREPARATION: imports, data loading, cleaning
******************************************** 
"""


# Package imports
import numpy as np
import scipy as scp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from cycler import cycler
from scipy import stats

warnings.filterwarnings("ignore")


# Reading dataset
train_file_path = "./input/master.csv"
dataset_df = pd.read_csv(train_file_path)
# print("Loading dataset CSV...")
# print("Full dataset shape is {}\n".format(dataset_df.shape))
# print("Here's the first few rows:\n")
# print(dataset_df.head(3))
# print()

num_countries = len(dataset_df["country"].unique())
num_age_groups = len(dataset_df["age"].unique())
first_year = min(dataset_df["year"])
last_year = max(dataset_df["year"])
# print("There are {} countries listed, with {} different age categories.\n".format(num_countries, num_age_groups))
# print("The data set covers the years {} to {}.\n".format(first_year, last_year))

# Cleaning up data:
dataset_df = dataset_df.rename(columns={'suicides/100k pop': 'suicides/100k',
                                        'HDI for year': 'HDI_for_year',
                                        ' gdp_for_year ($) ': 'gdp_for_year', 
                                        'gdp_per_capita ($)': 'gdp_per_capita'})
# print("The column headings are: {}".format(dataset_df.columns))

dataset_df['gdp_for_year'] = dataset_df['gdp_for_year'].replace(',','', regex=True)
dataset_df['gdp_for_year'] = pd.to_numeric(dataset_df['gdp_for_year'])


'''
******************
DATA VISUALIZATION
******************
'''

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

def calculate_avg_suicide(country_year):
    """Calculates the total number of suicides per 100k population for a given country in a given year.

    Args:
        country_year (str): A country/year identification string from the original dataset.

    Returns:
        [country, year, pop, avg] where 'country' and 'year' are the separated parts of the identifier, 'pop' is the population for that year
        and 'avg' is the total suicides per 100k population.
    """    
    if country_year == None:
        print("No argument passed to calculate_avg_suicide!")
    cysubset = dataset_df[dataset_df['country-year'] == country_year]
    country = cysubset['country'].loc[cysubset.index[0]]
    year = cysubset['year'].loc[cysubset.index[0]]
    pop = sum(cysubset['population'])
    avg = (sum(cysubset['suicides_no']) / pop) * 100000
    return [country, year, pop, avg]

def multiplot(dataset=dataset_df, x='', y='', z='', xlabel='', ylabel='', title='', figsize=(40,40), filename=''):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylim(min(dataset[y]), max(dataset[y]))
    ax.set_xlim(min(dataset[x]), max(dataset[x]))
    for i, w in enumerate(dataset[z].drop_duplicates()):
        select_z = dataset[dataset[z] == w]
        ax.plot(select_z[x], select_z[y], label=str(w), linewidth=2)
    ax.legend()
    improve_legend(ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig('./output/'+filename+'.png')



# First, amount of data by country:
plt.figure(figsize=(10,20))
sns.countplot(y='country', data=dataset_df, alpha=0.7)
plt.title("Amount of data by country")
plt.ylabel("Country")
plt.xlabel("Amount of data points")
plt.tight_layout()
plt.savefig("./output/data_by_country.png")
print("Data by country plot complete.")

plt.rcParams.update({'font.size': 22})

# Now GDP by country

country_gdp_years = dataset_df[["country", "year", "gdp_for_year"]].drop_duplicates()
multiplot(country_gdp_years, 'year', 'gdp_for_year', 'country', 'Year', 'GDP in USD', 'GDP for each country over time', (40,40), 'GDP_by_country')
print("GDP by country plot complete.")

# Redo of the above, but remove the top few countries to see the bottom end more clearly.
trimmed_country_gdp_years = country_gdp_years[(country_gdp_years['country'] != 'United States') & 
                                              (country_gdp_years['country'] != 'France') &
                                              (country_gdp_years['country'] != 'United Kingdom') &
                                              (country_gdp_years['country'] != 'Germany') &
                                              (country_gdp_years['country'] != 'Japan')]
multiplot(trimmed_country_gdp_years, 'year', 'gdp_for_year', 'country', 'Year', 'GDP in USD', 'Trimmed Countries GDP over time', (40,40), 'GDP_by_country_trimmed')
print("Trimmed GDP by country plot complete.")

# Now visualization of suicide rates (total no. of suicides / 100k pop) for each country: 

avg_suicides = []
for country_year in dataset_df['country-year'].drop_duplicates():
    avg_suicides.append(calculate_avg_suicide(country_year))

avg_suicides_df = pd.DataFrame(avg_suicides, columns=['country', 'year', 'population', 'suicides/100k'])
avg_suicides_df_compact = avg_suicides_df.drop('population', axis=1)
multiplot(avg_suicides_df_compact, 'year', 'suicides/100k', 'country', 'Year', 'Suicides per 100k population', 'Suicide rates in each country over time', (40,40), 'suicides_by_country')
print("Suicides/100k by country plot complete.")

# What about the average suicide rate for each country?

def calc_mean_suicide_rate(country=None):
    if country == None:
        print("Unable to calculate mean suicide rate - no country given.")
    else:
        result = 0
        country_rate_data = pd.DataFrame(avg_suicides_df[avg_suicides_df['country'] == country])
        total_pop = sum(country_rate_data['population'])
        num_country_years = len(country_rate_data['year'])
        for i in range(num_country_years):
            country_year_pop = country_rate_data.iat[i, 2]
            country_year_rate = country_rate_data.iat[i, 3]
            result += ((country_year_pop / total_pop) * country_year_rate)
        return result

            
mean_suicide_rates = []
for country in dataset_df['country'].drop_duplicates():
    avg_rate = calc_mean_suicide_rate(country)
    mean_suicide_rates.append([country, avg_rate])
mean_suicide_rates_df = pd.DataFrame(mean_suicide_rates, columns=['country', 'avg_suicides/100k'])
# print(mean_suicide_rates_df.head())

fig1, ax1 = plt.subplots(figsize=(40,40))
y_pos = np.arange(len(mean_suicide_rates_df['country']))
ax1.barh(y_pos, mean_suicide_rates_df['avg_suicides/100k'], align='center')
ax1.set_yticks(y_pos, labels = mean_suicide_rates_df['country'])
ax1.invert_yaxis()
ax1.set_xlabel('Weighted average of suicides per 100k')
ax1.set_title('Average suicide rate in each country')
fig1.tight_layout()
fig1.savefig('./output/avg_suicide_rate_by_country.png')
print('Average suicide rate by country plot complete.')


# Linear regression time! First, GDP against suicide rates... with one country?
def get_gdp_with_rates(country=None):
    if country == None:
        print("Unable to retrieve GDP vs. suicide rates. No country given.")
    else:
        country_rate_data = avg_suicides_df_compact[avg_suicides_df_compact['country'] == country]
        country_gdp_data = country_gdp_years[country_gdp_years['country'] == country]
        return country_rate_data['suicides/100k'], country_gdp_data['gdp_for_year']

irish_suicides, irish_gdp = get_gdp_with_rates('Ireland')
# print("There are {} data points for Irish suicide rates, and {} data points for Irish GDP.".format(len(irish_suicides), len(irish_gdp)))
slope, intercept, r, p, std_err = stats.linregress(irish_gdp, irish_suicides)

def predict_suicides_by_gdp(x):
    return slope * x + intercept

fig2, ax2 = plt.subplots(figsize=(40,40))
ax2.scatter(irish_gdp, irish_suicides)
fig2.savefig('./output/gdp_vs_suicides_ireland.png')