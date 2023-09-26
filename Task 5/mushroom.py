import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    column_names = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing",
                    "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                    "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
                    "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
    return pd.read_csv(url, names=column_names)

def explore_habitat_distribution(data):
    habitat_distribution = pd.crosstab(data['habitat'], data['class'])
    print(habitat_distribution)

def convert_to_dummies(data):
    data_dummies = pd.get_dummies(data, drop_first=True)  # drop_first for å unngå dummy variable trap
    print(data_dummies.head())  # Viser topp 5 rader av datasettet etter konvertering
    return data_dummies  # Viser topp 5 rader av datasettet etter konvertering


def visualize_with_tsne(data_dummies):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(data_dummies)

    data_dummies['tsne-2d-one'] = tsne_results[:, 0]
    data_dummies['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='tsne-2d-one', y='tsne-2d-two',
        hue='class_p',
        data=data_dummies,
        palette={True: "red", False: "green"},
        alpha=0.8
    )
    plt.title('2D t-SNE visualization of feature space')

    # Adding custom legend
    legend_labels = ['Edible (False)', 'Poisonous (True)']
    plt.legend(title='Class', labels=legend_labels)

    plt.show()


def main():
    data = load_data()
    explore_habitat_distribution(data)
    data_dummies = convert_to_dummies(data)
    visualize_with_tsne(data_dummies)


if __name__ == '__main__':
    main()
