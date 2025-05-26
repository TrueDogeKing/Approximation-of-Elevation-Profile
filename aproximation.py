import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
import pandas as pd
import os
# === Wczytywanie danych wysokościowych z pliku CSV ===
def load_elevation_data(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline()
    has_header = any(c.isalpha() for c in first_line)

    if has_header:
        data = pd.read_csv(file_path)
        data = data.iloc[:, :2]
        data.columns = ['distance', 'elevation']
    else:
        data = pd.read_csv(file_path, header=None)
        data.columns = ['distance', 'elevation']

    return data['distance'].values, data['elevation'].values


def scale_to_unit_interval(x, a, b):
    return 2 * (x - a) / (b - a) - 1  # do [-1, 1]

def rescale_from_unit_interval(x_scaled, a, b):
    return (x_scaled + 1) * (b - a) / 2 + a  # z [-1, 1] do [a, b]



# === Interpolacja Lagrange’a ===
def interpolate_with_lagrange(distances, elevations, query_points):
    a, b = distances[0], distances[-1]

    # Skaluj dziedzinę
    scaled_distances = scale_to_unit_interval(distances, a, b)
    scaled_query = scale_to_unit_interval(query_points, a, b)

    # Interpolacja na przeskalowanej dziedzinie
    polynomial = lagrange(scaled_distances, elevations)
    interpolated_scaled = polynomial(scaled_query)

    return interpolated_scaled


# === Interpolacja funkcjami sklejanymi (Cubic Spline) ===
def interpolate_with_cubic_spline(distances, elevations, query_points):
    spline = CubicSpline(distances, elevations)
    return spline(query_points)

def chebyshev_nodes(a, b, n):
    k = np.arange(n)
    x_cheb = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k + 1) * np.pi / (2 * n))
    return np.sort(x_cheb)  # sortujemy rosnąco dla czytelności


# === Wykres pojedynczej interpolacji ===
def plot_single_interpolation(original_distances, original_elevations,
                              node_distances, node_elevations,
                              query_distances, interpolated_values,
                              title='',save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(original_distances, original_elevations, '-', label='Dane oryginalne', color='gray')
    plt.plot(node_distances, node_elevations, 'o', label='Węzły interpolacji', color='red')
    plt.plot(query_distances, interpolated_values, label='Interpolacja', color='blue')
    plt.xlabel('Odległość (m)')
    plt.ylabel('Wysokość (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    #plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.close()  # zamiast show() – nie otwiera wykresu w o

# === Wykonywanie analizy interpolacji dla różnych liczby węzłów ===
def run_interpolation_with_varying_nodes(file_path, method='lagrange', title_prefix='', chebyshev=False, output_dir=None):
    distances, elevations = load_elevation_data(file_path)
    query_distances = np.linspace(distances[0], distances[-1], 300)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for n_nodes in [5, 16, 52, 103]:
        if chebyshev:
            node_distances = chebyshev_nodes(distances[0], distances[-1], n_nodes)
            node_elevations = np.interp(node_distances, distances, elevations)
        else:
            subset_indices = np.linspace(0, len(distances) - 1, n_nodes, dtype=int)
            node_distances = distances[subset_indices]
            node_elevations = elevations[subset_indices]

        # Domyślne wartości
        interp_label = ''
        file_suffix = ''

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


run_interpolation_with_varying_nodes("2018_paths/SpacerniakGdansk.csv", method='lagrange', title_prefix="Trasa 3", output_dir="plots/trasa3")
run_interpolation_with_varying_nodes("2018_paths/SpacerniakGdansk.csv", method='lagrange', title_prefix="Trasa 3", chebyshev=True, output_dir="plots/trasa3")
run_interpolation_with_varying_nodes("2018_paths/SpacerniakGdansk.csv", method='spline', title_prefix="Trasa 3", output_dir="plots/trasa3")

