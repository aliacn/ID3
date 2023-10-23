from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Verileri hazırlayın (örnek veri)
X = [
    ['Erkek', 'Genç', 'Hafif'],
    ['Erkek', 'Orta', 'Hafif'],
    ['Erkek', 'Genç', 'Ağır'],
    ['Kadın', 'Yaşlı', 'Hafif'],
    ['Kadın', 'Orta', 'Ağır']
]

y = ['Negatif', 'Negatif', 'Negatif', 'Pozitif', 'Pozitif']

# Label encoding ile cinsiyet, yaş, kilo ve teşhis özelliklerini dönüştürün
label_encoder = LabelEncoder()
X_encoded = []
for i in range(len(X[0])):
    X_encoded.append(label_encoder.fit_transform([x[i] for x in X]))

# Karar ağacı modelini oluşturun ve eğitin
model = DecisionTreeClassifier()
model.fit(list(zip(*X_encoded)), y)

# Tahmin yapalım Kadın - Genç -
tahmin = model.predict([[1, 0, 2]])
print("Tahmin:", tahmin)
