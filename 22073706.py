import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import cluster_tools as ct
import scipy.optimize as opt
import errors as err

def read_and_clean_data(file_path):
    """
    Reads the data from the given file, cleans it, and returns cleaned DataFrames.

    Parameters
    ----------
    file_path : str
        The file path to be read into a DataFrame.

    Returns
    -------
    cleaned_df : pandas DataFrame
        The cleaned version of the ingested DataFrame.
    transposed_df : pandas DataFrame
        The transposed version of the cleaned DataFrame.

    """
    if ".xlsx" in file_path:
        data_df = pd.read_excel(file_path, index_col=0)
    elif ".csv" in file_path:
        data_df = pd.read_csv(file_path, index_col=0)
    else:
        print("Invalid filetype")
        return None, None

    cleaned_df = data_df.dropna(axis=1, how="all").dropna()
    transposed_df = cleaned_df.transpose()

    return cleaned_df, transposed_df

# For reproducibility
np.random.seed(10)

def perform_kmeans_clustering(num_clusters, data):
    """
    Performs k-means clustering on the given data and returns cluster labels and centers.

    Parameters
    ----------
    num_clusters : int
        The number of clusters.
    data : pandas DataFrame
        The DataFrame on which clustering is performed.

    Returns
    -------
    labels : ndarray
        The labels of the clusters.
    centers : ndarray
        The coordinates of the cluster centers.

    """
    kmeans = cluster.KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    return labels, centers

def polynomial_curve(x, a, b, c):
    """
    Generates a polynomial curve for fitting the data.

    Parameters
    ----------
    x : int or float
        The variable of the polynomial.
    a : int or float
        The constant of the polynomial.
    b : int or float
        The coefficient of x.
    c : int or float
        The coefficient of x**2.

    Returns
    -------
    f : ndarray
        The polynomial curve.

    """
    x = x - 2003
    f = a + b * x + c * x**2

    return f

# The CSV files are read into DataFrames
_, co2_data = read_and_clean_data("co2_emissions.csv")
print(co2_data)

_, gdp_data = read_and_clean_data("gdp_per_capita.csv")
print(gdp_data)

# Specific columns are extracted
co2_finland = co2_data.loc[:, "Finland"].copy()
print(co2_finland)

gdp_per_capita_finland = gdp_data.loc["1990":"2019", "Finland"].copy()
print(gdp_per_capita_finland)

# The extracted columns are merged into a DataFrame
finland_df = pd.merge(co2_finland, gdp_per_capita_finland, on=co2_finland.index, how="outer")
finland_df = finland_df.rename(columns={'key_0': "Year", 'Finland_x': "co2_emissions", 'Finland_y': "gdp_per_capita"})
finland_df = finland_df.set_index("Year")
print(finland_df)

# The scatter matrix of the DataFrame is plotted
pd.plotting.scatter_matrix(finland_df)

# The DataFrame for clustering is created
cluster_df = finland_df[["co2_emissions", "gdp_per_capita"]].copy()

# The data is normalized
cluster_df, min_values, max_values = ct.scaler(cluster_df)

# The number of clusters and respective silhouette scores are printed
print("n   score")
for num_clusters in range(2, 10):
    labels, centroids = perform_kmeans_clustering(num_clusters, cluster_df)
    print(num_clusters, skmet.silhouette_score(cluster_df, labels))

# The cluster centers and labels are calculated using the function
cluster_labels, cluster_centers = perform_kmeans_clustering(5, cluster_df)
xcen = cluster_centers[:, 0]
ycen = cluster_centers[:, 1]

# The clustering is plotted
plt.figure()
color_map = plt.cm.get_cmap('Set1')
plt.scatter(cluster_df['gdp_per_capita'], cluster_df["co2_emissions"], s=10,
            c=cluster_labels, marker='o', cmap=color_map)
plt.scatter(xcen, ycen, s=20, c="k", marker="d")
plt.title("CO2 emission vs GDP per capita of Finland", fontsize=20)
plt.xlabel("GDP per capita", fontsize=18)
plt.ylabel("CO2 emissions", fontsize=18)
plt.show()

# The cluster centers are rescaled to the original scale
rescaled_centers = ct.backscale(cluster_centers, min_values, max_values)
xcen = rescaled_centers[:, 0]
ycen = rescaled_centers[:, 1]

# The clustering is plotted with the original scale
plt.figure()
color_map = plt.cm.get_cmap('Set1')
plt.scatter(finland_df['gdp_per_capita'], finland_df["co2_emissions"], 10,
            cluster_labels, marker='o', cmap=color_map)
plt.xlabel("GDP per capita")
plt.ylabel("CO2 emissions")
plt.title("CO2 emission vs GDP per capita of Finland")
plt.show()

years = ['1990', '1995', '2000', '2005', '2010', '2015', '2020']

# The plot of CO2 Emissions (1990-2019) in Finland is plotted
plt.plot(finland_df.index, finland_df['co2_emissions'], color='blue')
plt.xlabel("Years", fontsize=16)
plt.ylabel("CO2 Emissions (metric tons per capita)", fontsize=12)
plt.title("CO2 Emissions (1990-2019)", fontsize=18)
plt.xticks(ticks=years, labels=years)
plt.show()

# The plot of GDP per capita (1990-2019) in Finland is plotted
plt.plot(finland_df.index, finland_df["gdp_per_capita"], color='green')
plt.xlabel("Years", fontsize=16)
plt.ylabel("GDP per capita", fontsize=14)
plt.title("GDP per capita (1990-2019)", fontsize=18)
plt.xticks(ticks=years, labels=years)
plt.show()

# The DataFrame is prepared for fitting
finland_df = finland_df.reset_index()
finland_df["gdp_per_capita"] = pd.to_numeric(finland_df["gdp_per_capita"])
finland_df["Year"] = pd.to_numeric(finland_df["Year"])

# The fitting of the GDP per capita plot
# Calculates the parameters and covariance
params, covariance = opt.curve_fit(polynomial_curve, finland_df["Year"],
                                   finland_df["gdp_per_capita"])
# Calculates the standard deviation
sigma_values = np.sqrt(np.diag(covariance))
forecast_years = np.arange(1990, 2030)
# Calculates the fitting curve
gdp_forecast = polynomial_curve(forecast_years, *params)
# Calculates the confidence range
lower_bound, upper_bound = err.err_ranges(forecast_years, polynomial_curve, params, sigma_values)
finland_df["fit1"] = polynomial_curve(finland_df["Year"], *params)
# Plots the graph with fitting and confidence range
plt.figure()
plt.plot(finland_df["Year"], finland_df["gdp_per_capita"], label="GDP", color='blue')
plt.plot(forecast_years, gdp_forecast, label="forecast", color='green')
plt.fill_between(forecast_years, lower_bound, upper_bound, color="skyblue", alpha=0.8)  # Change "yellow" to "skyblue"
plt.xlabel("Year", fontsize=16)
plt.ylabel("GDP per capita", fontsize=14)
plt.title("GDP per capita forecast of Finland", fontsize=18)
plt.legend()
plt.show()


# The fitting of CO2 Emissions plot
# Calculates the parameters and covariance
params, covariance = opt.curve_fit(polynomial_curve, finland_df["Year"], finland_df["co2_emissions"])
# Calculates the standard deviation
sigma_values = np.sqrt(np.diag(covariance))
# Calculates the fitting curve
co2_forecast = polynomial_curve(forecast_years, *params)
# Calculates the confidence range
lower_bound, upper_bound = err.err_ranges(forecast_years, polynomial_curve, params, sigma_values)
finland_df["fit2"] = polynomial_curve(finland_df["Year"], *params)
# Plots the graph with fitting and confidence range
plt.figure()
plt.plot(finland_df["Year"], finland_df["co2_emissions"], label="CO2 emissions", color='blue')
plt.plot(forecast_years, co2_forecast, label="forecast", color="green")
plt.fill_between(forecast_years, lower_bound, upper_bound, color="skyblue", alpha=0.8)  # Change "yellow" to "skyblue"
plt.xlabel("Year", fontsize=16)
plt.ylabel("CO2 Emissions (metric tons per capita)", fontsize=12)
plt.title("CO2 Emissions Forecast of Finland", fontsize=18)
plt.legend()
plt.show()

# The DataFrame is prepared for fitting
finland_df = finland_df.reset_index()
finland_df["gdp_per_capita"] = pd.to_numeric(finland_df["gdp_per_capita"])
finland_df["Year"] = pd.to_numeric(finland_df["Year"])

# The fitting of the GDP per capita plot
# Calculates the parameters and covariance
params, covariance = opt.curve_fit(polynomial_curve, finland_df["Year"],
                                   finland_df["gdp_per_capita"])
# Calculates the standard deviation
sigma_values = np.sqrt(np.diag(covariance))
forecast_years = np.arange(1990, 2030)
# Calculates the fitting curve
gdp_forecast = polynomial_curve(forecast_years, *params)
# Calculates the confidence range
lower_bound, upper_bound = err.err_ranges(forecast_years, polynomial_curve, params, sigma_values)
finland_df["fit1"] = polynomial_curve(finland_df["Year"], *params)

# Pie chart for GDP distribution (example data, replace with actual data)
gdp_distribution = [60, 30, 10]  # Example distribution, adjust as needed
fig, ax3 = plt.subplots(figsize=(6, 6))  # Adjust the figure size if needed
ax3.pie(gdp_distribution, labels=["Industry", "Services", "Agriculture"], autopct='%1.1f%%', startangle=90, colors=['orange', 'lightcoral', 'lightgreen'])
ax3.set_title("GDP Distribution of Finland", fontsize=14)
plt.show()

# The DataFrame is prepared for fitting
finland_df = finland_df.reset_index()
finland_df["gdp_per_capita"] = pd.to_numeric(finland_df["gdp_per_capita"])
finland_df["Year"] = pd.to_numeric(finland_df["Year"])

# The fitting of the GDP per capita plot
# Calculates the parameters and covariance
params, covariance = opt.curve_fit(polynomial_curve, finland_df["Year"],
                                   finland_df["gdp_per_capita"])
# Calculates the standard deviation
sigma_values = np.sqrt(np.diag(covariance))
forecast_years = np.arange(1990, 2051)  # Extended to 2050
# Calculates the fitting curve
gdp_forecast = polynomial_curve(forecast_years, *params)
# Calculates the confidence range
lower_bound, upper_bound = err.err_ranges(forecast_years, polynomial_curve, params, sigma_values)
finland_df["fit1"] = polynomial_curve(finland_df["Year"], *params)

# Line plot for GDP per capita
plt.figure(figsize=(10, 6))
plt.plot(finland_df["Year"], finland_df["gdp_per_capita"], label="GDP", color='blue')
plt.plot(forecast_years, gdp_forecast, label="forecast", color='red')
plt.fill_between(forecast_years, lower_bound, upper_bound, color="skyblue", alpha=0.8)
plt.xlabel("Year", fontsize=16)
plt.ylabel("GDP per capita", fontsize=14, color='blue')
plt.title("GDP per capita forecast of Finland", fontsize=18)
plt.legend()
plt.show()
