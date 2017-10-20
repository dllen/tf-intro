import matplotlib.pyplot as plt
import loadData
import os

ROOT_PATH = "../../"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = loadData.load_data(train_data_directory)

unique_labels = set(labels)

plt.figure(figsize=(15, 15))

i = 1

for label in unique_labels:
    image = images[labels.index(label)]
    plt.subplot(8, 8, i)
    plt.axis('off')
    plt.title("label {0} ({1})".format(label, labels.count(label)))
    i += 1
    plt.imshow(image)

plt.show()
