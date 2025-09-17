from scipy.stats import uniform, norm, probplot
import numpy as np
import matplotlib.pyplot as plt

## EXERCISE 1

# -------------------------------- PART (A) -----------------------------------
# --- Parameters ---
# Sample sizes
sample_sizes = [45, 100, 10**3, 10**5] 

# Uniform distribution parameters
loc = 0
scale = 1

# --- Plot 1: theoretical cdf v. empirical cdf ---

plt.figure(figsize=(8, 7))

# 1. Theoretical cdf
x_values = np.linspace(loc, loc + scale, 1000)
cdf_theoretical = uniform.cdf(x_values, loc=loc, scale=scale)
plt.plot(x_values, cdf_theoretical, "k-", lw=2, label="Theoretical cdf of U(0, 1)")

# 2. Empirical cdf
for n in sample_sizes:
    # Generate n random numbers from U(0,1)
    random_data = uniform.rvs(loc=loc, scale=scale, size=n)
    
    # For empirical cdf, we sort x data (random numbers) and create an equispaced y-axis scaled from 1/n to 1
    x_ecdf = np.sort(random_data)
    y_ecdf = np.arange(1, n + 1) / n
    
    plt.plot(x_ecdf, y_ecdf, marker='.', linestyle='none', ms=2, label=f"ECDF (n={n})")

# Plot configuration
plt.title("Theoretical cdf v. Empirical cdf of $\mathcal{U}(0,1)$")
plt.xlabel("x-values")
plt.ylabel("cdf(x)")
plt.grid(True)
plt.legend()
plt.show()

# --- Plot 2: Q-Q plot ---

# Crear una figura con una cuadrícula de 2x2 para los gráficos Q-Q
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# Aplanar el array de ejes para iterar fácilmente sobre él
axes = axes.flatten()

for i, n in enumerate(sample_sizes):
    # Generar nuevos datos aleatorios para el análisis Q-Q
    random_data = uniform.rvs(loc=loc, scale=scale, size=n)
    
    # Crear el gráfico Q-Q comparando los datos con la distribución teórica U(0,1)
    probplot(random_data, dist=uniform, plot=axes[i])
    
    # Configuraciones de cada subgráfico
    axes[i].set_title(f"Q-Q plot (n={n})")
    axes[i].set_xlabel("Theoretical quantiles $U(0,1)$")
    axes[i].set_ylabel("Empirical quantiles")

# Ajustar el espaciado
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------
# -------------------------------- PART (B) -----------------------------------
alpha = 0.1

for i in sample_sizes:
    data = uniform.rvs(loc=loc, scale=scale, size=i)
    sorted_data = np.sort(data)

    F_theoretical = uniform.cdf(sorted_data)
    F_empirical = np.arange(1, i+1)/i

    D_plus = np.max(F_empirical-F_theoretical)
    D_minus = np.max(F_theoretical - (np.arange(0,n)/n))
    D_n = max(D_minus,D_plus)

    scaled_Dn = np.sqrt(i) * D_n