import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
import pandas as pd

# Load elevation data from a CSV file
def load_elevation_data(file_path):
    # Check if first line contains any letters (header or not)
    with open(file_path, 'r') as f:
        first_line = f.readline()
    
    has_header = any(c.isalpha() for c in first_line)

    if has_header:
        # Load with header, then ignore column names and assign your own
        data = pd.read_csv(file_path)
        data = data.iloc[:, :2]  # take first two columns only
        data.columns = ['distance', 'elevation']
    else:
        # No header, just load and assign columns
        data = pd.read_csv(file_path, header=None)
        data.columns = ['distance', 'elevation']

    return data['distance'].values, data['elevation'].values



# Lagrange Interpolation
def interpolate_with_lagrange(distances, elevations, query_points):
    polynomial = lagrange(distances, elevations)
    return polynomial(query_points)

# Cubic Spline Interpolation
def interpolate_with_cubic_spline(distances, elevations, query_points):
    spline = CubicSpline(distances, elevations)
    return spline(query_points)

# Plot results
def plot_profiles(original_distances, original_elevations, query_distances, lagrange_result, spline_result, title='Elevation Profile Approximation'):
    plt.figure(figsize=(10, 6))
    plt.plot(original_distances, original_elevations, 'o', label='Original Data', color='black')
    plt.plot(query_distances, lagrange_result, '-', label='Lagrange Interpolation', color='blue')
    plt.plot(query_distances, spline_result, '--', label='Cubic Spline Interpolation', color='green')
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to run the interpolation comparison
def run_interpolation_comparison(file_path):
    # Load the full dataset
    distances, elevations = load_elevation_data(file_path)

    # Select a subset of points for interpolation
    subset_indices = np.linspace(0, len(distances) - 1, 10, dtype=int)
    subset_distances = distances[subset_indices]
    subset_elevations = elevations[subset_indices]

    # Define query points
    query_distances = np.linspace(distances[0], distances[-1], 300)

    # Apply interpolation methods
    lagrange_values = interpolate_with_lagrange(subset_distances, subset_elevations, query_distances)
    spline_values = interpolate_with_cubic_spline(subset_distances, subset_elevations, query_distances)

    # Plot the results
    plot_profiles(distances, elevations, query_distances, lagrange_values, spline_values)

# Example use
run_interpolation_comparison("2018_paths/WielkiKanionKolorado.csv")

