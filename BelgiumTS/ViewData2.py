import os
import loadData
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray

ROOT_PATH = "../../"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = loadData.load_data(train_data_directory)

images28 = [transform.resize(image, (28, 28)) for image in images]

images28 = np.array(images28)

images28 = rgb2gray(images28)

traffic_signs = [300, 2250, 3650, 4000]

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i + 1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)

plt.show()