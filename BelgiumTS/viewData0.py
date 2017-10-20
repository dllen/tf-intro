import os
import loadData
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

ROOT_PATH = "../../"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = loadData.load_data(train_data_directory)

print(np.array(images).ndim)

print(np.array(images).size)

print(len(set(labels)))

# plt.hist(labels, 62)
# plt.show()

traffic_signs = [300, 2250, 3650, 4000]

for i in range(len(traffic_signs)):
    plt.subplots_adjust(wspace=0.5)
    image = images[traffic_signs[i]]
    print("shape: {0}, min: {1}, max: {2}".format(image.shape,
                                                  image.min(),
                                                  image.max()))
    image28 = transform.resize(image, (28, 28))
    print("shape: {0}, min: {1}, max: {2}".format(image28.shape,
                                                  image28.min(),
                                                  image28.max()))
    plt.subplot(1, 4, i + 1)
    plt.axis('off')
    plt.imshow(image)
    plt.subplot(2, 4, i + 1)
    plt.axis('off')
    plt.imshow(image28)

plt.show()
