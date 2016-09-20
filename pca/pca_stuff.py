import numpy as np
from keras.datasets import cifar10
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches


(Xtrain, ytrain), (Xtest, ytest) = cifar10.load_data()
rng = np.random.RandomState(42)
perm = rng.permutation(Xtrain.shape[0])
n_images = 1000

patches_train = extract_patches(Xtrain[perm[:n_images]],
        (1, 3, 5, 5), (1, 1, 1, 1))
patches_reshaped = patches_train.reshape(n_images * 28 * 28, 3 * 5 * 5)

pca1 = PCA()
transformed1 = pca1.fit_transform(patches_reshaped)

realigned = transformed1.reshape(n_images, 28, 28, 75)

patches2 = extract_patches(realigned, (1, 5, 5, 1), (1, 1, 1, 1))
patches2_reshaped = patches2.reshape(n_images * 24 * 24 * 75, 5 * 5)

pca2 = PCA()
transformed2 = pca2.fit_transform(patches2_reshaped)

realigned2 = transformed2.reshape()






import matplotlib.pyplot as plt
plt.figure()

for i in range(0, 75, 5):
    plt.subplot(1, 75 // 5 + 1, i // 5 + 2)
    plt.imshow(realigned[:15, :, :, i].reshape(-1, 28))
    plt.axis('off')
    plt.gray()

plt.subplot(1, 75 // 5 + 1, 1)
plt.imshow(Xtrain[perm[:15]].transpose(0, 2, 3, 1).reshape(-1, 32, 3))

plt.figure()
padded = np.pad(pca1.components_.reshape(75, 3, 5, 5),
        ((0, 0), (0,0),(2, 2), (2, 2)), mode='constant')
plt.imshow(padded.reshape(5, 15, 3, 9, 9).transpose(
    0, 3, 1, 4, 2).reshape(45, 135, 3))


plt.show()





