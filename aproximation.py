import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# === Load elevation data from CSV ===
def load_elevation_data(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline()
    has_header = any(char.isalpha() for char in first_line)

    if has_header:
        data = pd.read_csv(file_path)
        data = data.iloc[:, :2]
        data.columns = ['distance', 'elevation']
    else:
        data = pd.read_csv(file_path, header=None)
        data.columns = ['distance', 'elevation']

    return data['distance'].values, data['elevation'].values

# === Scaling functions ===
def scale_to_unit_interval(x, a, b):
    return 2 * (x - a) / (b - a) - 1

def rescale_from_unit_interval(x_scaled, a, b):
    return (x_scaled + 1) * (b - a) / 2 + a

# === Manual Lagrange interpolation ===
def interpolate_with_lagrange(x_nodes, y_nodes, x_query):
    def lagrange_basis(k, x):
        terms = [(x - x_nodes[j]) / (x_nodes[k] - x_nodes[j])
                 for j in range(len(x_nodes)) if j != k]
        return np.prod(terms, axis=0)

    y_query = np.zeros_like(x_query)
    for k in range(len(x_nodes)):
        Lk = np.array([lagrange_basis(k, x) for x in x_query])
        y_query += y_nodes[k] * Lk
    return y_query

# === Manual natural cubic spline interpolation ===
def interpolate_with_cubic_spline(x, y, x_query):
    n = len(x)
    h = np.diff(x)
    alpha = [0] + [3/h[i]*(y[i+1]-y[i]) - 3/h[i-1]*(y[i]-y[i-1]) for i in range(1, n-1)]

    l = [1] + [0]*(n-1)
    mu = [0]*(n-1) + [0]
    z = [0]*(n)

    for i in range(1, n-1):
        l_i = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        l.append(l_i)
        mu[i] = h[i]/l_i
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l_i

    l.append(1)
    z[n-1] = 0
    c = [0]*n
    b = [0]*(n-1)
    d = [0]*(n-1)
    a = list(y)

    for j in reversed(range(n-1)):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (a[j+1] - a[j])/h[j] - h[j]*(c[j+1] + 2*c[j])/3
        d[j] = (c[j+1] - c[j]) / (3*h[j])

    result = []
    for xq in x_query:
        for i in range(n - 1):
            if x[i] <= xq <= x[i+1]:
                dx = xq - x[i]
                val = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
                result.append(val)
                break
    return np.array(result)

# === Chebyshev nodes ===
def optimal_chebyshev_nodes(a, b, n):
    return sorted([
        0.5*(a + b) + 0.5*(b - a)*np.cos((2*k - 1)*np.pi / (2*n))
        for k in range(1, n+1)
    ])

# === Plotting ===
def plot_single_interpolation(original_distances, original_elevations,
                              node_distances, node_elevations,
                              query_distances, interpolated_values,
                              title='', save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(original_distances, original_elevations, '-', label='Dane oryginalne', color='gray')
    plt.plot(node_distances, node_elevations, 'o', label='Węzły interpolacji', color='red')
    plt.plot(query_distances, interpolated_values, label='Interpolacja', color='blue')
    plt.xlabel('Odległość (m)')
    plt.ylabel('Wysokość (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# === Full analysis ===
def run_interpolation_with_varying_nodes(file_path, method='lagrange', title_prefix='',
                                         chebyshev=False, output_dir=None):
    distances, elevations = load_elevation_data(file_path)
    query_distances = np.linspace(distances[0], distances[-1], 300)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for n_nodes in [5, 16, 52, 103]:
        if chebyshev:
            node_distances = optimal_chebyshev_nodes(distances[0], distances[-1], n_nodes)
            node_elevations = np.interp(node_distances, distances, elevations)
        else:
            subset_indices = np.linspace(0, len(distances) - 1, n_nodes, dtype=int)
            node_distances = distances[subset_indices]
            node_elevations = elevations[subset_indices]

        if method == 'lagrange':
            interpolated = interpolate_with_lagrange(node_distances, node_elevations, query_distances)
            interp_label = 'Lagrange (Czebyszew)' if chebyshev else 'Lagrange (równomierne)'
            file_suffix = 'lagrange_chebyshev' if chebyshev else 'lagrange_uniform'
        elif method == 'spline':
            interpolated = interpolate_with_cubic_spline(node_distances, node_elevations, query_distances)
            interp_label = 'Splajn trzeciego stopnia'
            file_suffix = 'spline'
        else:
            raise ValueError(f"Nieznana metoda interpolacji: {method}")

        title = f'{title_prefix} - {interp_label}, {n_nodes} węzłów'

        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{base_name}_{file_suffix}_{n_nodes}_wezly.png"
            save_path = os.path.join(output_dir, filename)

        plot_single_interpolation(
            distances, elevations,
            node_distances, node_elevations,
            query_distances, interpolated,
            title=title,
            save_path=save_path
        )

# === Run for various settings ===
run_interpolation_with_varying_nodes("2018_paths/SpacerniakGdansk.csv", method='lagrange',
                                     title_prefix="Trasa 3", output_dir="plots/trasa3")

run_interpolation_with_varying_nodes("2018_paths/SpacerniakGdansk.csv", method='lagrange',
                                     title_prefix="Trasa 3", chebyshev=True, output_dir="plots/trasa3")

run_interpolation_with_varying_nodes("2018_paths/SpacerniakGdansk.csv", method='spline',
                                     title_prefix="Trasa 3", output_dir="plots/trasa3")
