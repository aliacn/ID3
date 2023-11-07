import numpy as np

# Veri setini oluşturun (D1, D2, D3, Sınıf)
data = np.array([
    [0, 8, 2, 'Poz'],
    [2, 6, 8, 'Poz'],
    [2, 8, 10, 'Neg'],
    [7, 7, 1, 'Poz']
])

# Verileri X ve y olarak ayırın
X = data[:, :-1].astype(float)  # Veri özellikleri (D1, D2, D3)
y = data[:, -1]  # Veri sınıfları

# Verileri Z skoru ile normalize edin
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_normalized = (X - mean) / std

# Yeni örneği tanımlayın
new_example = np.array([9, 3, 7])

# K-NN algoritması için k değerini belirleyin (K = 2)
k = 2

# Mesafeleri hesaplayın
distances = np.linalg.norm(X_normalized - (new_example - mean) / std, axis=1)

# Mesafelere göre sıralayın ve en yakın K komşuyu bulun (K = 2)
nearest_indices = np.argsort(distances)[:k]
nearest_classes = y[nearest_indices]

# Ağırlıklı oylama ile sınıf tahmini yapın
unique_classes, class_counts = np.unique(nearest_classes, return_counts=True)
weighted_votes = class_counts / distances[nearest_indices]
predicted_class = unique_classes[np.argmax(weighted_votes)]

print(f"({new_example[0]}, {new_example[1]}, {new_example[2]}) örneği {predicted_class} sınıfına aittir.")
print(f"Weighted Voting Sonuçları:")
for i, cls in enumerate(unique_classes):
    print(f"Sınıf: {cls}, Weighted Vote: {weighted_votes[i]:.2f}")
