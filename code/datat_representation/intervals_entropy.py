def shannon_entropy_from_bins(data, k):
    hist, _ = np.histogram(data, bins=k)
    p = hist / np.sum(hist)
    return entropy(p, base=2)

data = y_data[['LN_IC50']]

k_values = np.arange(2, 50)
entropies = np.array([shannon_entropy_from_bins(data, k) for k in k_values])

slopes = np.diff(entropies) / np.diff(k_values)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, entropies, marker='o')
plt.title("Entropy vs Number of bins")
plt.xlabel("k (number of bins)")
plt.ylabel("Entropy")

plt.subplot(1, 2, 2)
plt.plot(k_values[:-1], slopes, marker='x', color='orange')
plt.title("Entropy slope")
plt.xlabel("k (number of bins)")
plt.ylabel("Slope (dH/dk)")

plt.tight_layout()
plt.grid(True)
plt.show()