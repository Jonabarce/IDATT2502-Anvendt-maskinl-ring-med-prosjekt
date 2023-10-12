import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Funksjon for å laste ned og forberede data
def load_data():
    # Definer datasett-URL og kolonnenavn
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    column_names = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                    "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                    "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]

    # Last ned datasettet fra URL-en
    data = pd.read_csv(url, names=column_names)

    # Konverter kategoriske data til numeriske data
    label_enc = LabelEncoder()
    for column in data.columns:
        data[column] = label_enc.fit_transform(data[column])

    return data


# Last ned data
data = load_data()

# Initialiser en liste for å lagre Silhouette Scores for hver k
sil_scores = []
# Definer k-verdier som skal testes (fra 2 til 30)
k_values = list(range(2, 31))

# Kjør k-means algoritmen med forskjellige k-verdier og beregn Silhouette Scores
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    labels = kmeans.labels_
    sil_scores.append(silhouette_score(data, labels))

# Plott Silhouette Scores mot antall klynger
plt.figure(figsize=(10, 6))
plt.plot(k_values, sil_scores, marker='o')
plt.xlabel('Antall klynger')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Antall klynger')
plt.show()

# Velg optimal k med høyeste Silhouette Score
opt_k = k_values[sil_scores.index(max(sil_scores))]
kmeans = KMeans(n_clusters=opt_k)
kmeans.fit(data)
labels = kmeans.labels_

# Bruk PCA for å redusere dimensjonene av dataene til 2D for visualisering
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Visualiser klyngene i 2D
plt.figure(figsize=(10, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title('2D PCA av soppdatasett')
plt.show()
