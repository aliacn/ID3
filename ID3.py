import numpy as np
from collections import Counter
import math


# Karar ağacı düğümlerini temsil eden sınıfı oluşturuyoruz
class Node:
    def __init__(self, data):
        self.data = data
        self.children = {}


# Entropi hesaplayan fonksiyon
def entropy(data):
    labels = [item[-1] for item in data]
    label_counts = Counter(labels)
    entropy = 0
    total_items = len(data)

    for label in label_counts:
        probability = label_counts[label] / total_items
        entropy -= probability * math.log2(probability)

    return entropy


# Bilgi kazandırdığını hesaplayan fonksiyon
def information_gain(data, feature_index):
    total_entropy = entropy(data)
    feature_values = [item[feature_index] for item in data]
    unique_values = set(feature_values)

    weighted_entropy = 0

    for value in unique_values:
        subset = [item for item in data if item[feature_index] == value]
        probability = len(subset) / len(data)
        weighted_entropy += probability * entropy(subset)

    return total_entropy - weighted_entropy


# En  bulan fonksiyon
def most_common_label(data):
    labels = [item[-1] for item in data]
    label_counts = Counter(labels)
    return label_counts.most_common(1)[0][0]


# Karar ağacını oluşturan ana işlev
def build_tree(data, features):
    if len(set(item[-1] for item in data)) == 1:
        return Node(data[0][-1])

    if len(features) == 0:
        return Node(most_common_label(data))

    best_feature_index = np.argmax([information_gain(data, feature_index) for feature_index in range(len(features))])
    best_feature = features[best_feature_index]

    root = Node(best_feature)
    unique_values = set(item[best_feature_index] for item in data)

    for value in unique_values:
        subset = [item[:best_feature_index] + item[best_feature_index + 1:] for item in data if
                  item[best_feature_index] == value]
        child_features = features[:best_feature_index] + features[best_feature_index + 1:]
        root.children[value] = build_tree(subset, child_features)

    return root


# Karar ağacını yazdırmak için kullanılan yardımcı fonksiyon
def print_tree(node, depth=0):
    if isinstance(node, Node):
        print("  " * depth + node.data)
        for child_value, child_node in node.children.items():
            print("  " * (depth + 1) + f"{child_value}:")
            print_tree(child_node, depth + 2)
    else:
        print("  " * depth + node)


# Veri seti
data = [
    ["Orta", "Yaşlı", "Erkek", "Evet"],
    ["İlk", "Genç", "Erkek", "Hayır"],
    ["Yüksek", "Orta", "Kadın", "Hayır"],
    ["Orta", "Orta", "Erkek", "Evet"],
    ["İlk", "Orta", "Erkek", "Evet"],
    ["Yüksek", "Yaşlı", "Kadın", "Evet"],
    ["İlk", "Genç", "Kadın", "Hayır"]
]

# Özellikler
features = ["Eğitim", "Yaş", "Cinsiyet"]

# Karar ağacını oluştur
root_node = build_tree(data, features)

# Karar ağacını yazdır
print("Karar Ağacı:")
print_tree(root_node)

# Entropi değerlerini hesapla ve yazdır
print("\nEntropi Değerleri:")
print("Kök Düğüm Entropisi:", entropy(data))
for feature_index, feature_name in enumerate(features):
    ig = information_gain(data, feature_index)
    print(f"Bilgi Kazanımı ({feature_name}): {ig}")



import matplotlib.pyplot as plt

def plot_tree(node, parent_name, graph, depth=0):
    if isinstance(node, Node):
        for child_value, child_node in node.children.items():
            child_name = f"{parent_name} -> {child_value}"
            graph.node(child_name, label=child_value)
            graph.edge(parent_name, child_name)
            plot_tree(child_node, child_name, graph, depth + 1)
    else:
        leaf_name = f"{parent_name} -> {node}"
        graph.node(leaf_name, label=node, shape='box')
        graph.edge(parent_name, leaf_name)

from graphviz import Digraph

# Graphviz ile karar ağacını görselleştirebiliriz
dot = Digraph(comment='Karar Ağacı')
dot.node(root_node.data)
plot_tree(root_node, root_node.data, dot)

# Ağacı görüntüleme
dot.render('karar_agaci', view=True)
