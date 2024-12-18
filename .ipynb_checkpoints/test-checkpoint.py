import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets, manifold

# Import MNIST data
data = datasets.fetch_openml(
    'mnist_784',
    version=1,
    return_X_y=True
)

pixel_values, targets = data
targets = targets.astype(int)

#check one of the data point, convert the 784(28*28) to the 2 dimensional numpy array
single_image = pixel_values.iloc[1, :].to_numpy().reshape(28, 28)

plt.imshow(single_image, cmap='gray')

# Creates the t-SNE transformation of the data
tsne = manifold.TSNE(n_components=2, random_state=42)

print("1111")
transformed_data = tsne.fit_transform(pixel_values.iloc[:100, :].to_numpy())
print("2222")
print(transformed_data.shape)