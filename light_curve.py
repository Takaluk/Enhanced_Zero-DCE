import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_brightness(image_path):
    image = Image.open(image_path)
    image = image.convert("L")  #그레이스케일

    image_data = np.array(image)

    brightness_values = image_data.flatten()

    histogram, bins = np.histogram(brightness_values, bins=256, range=[0, 256])

    plt.figure(figsize=(10, 5))
    plt.plot(histogram, color='black')
    plt.xlim([0, 256])
    plt.ylim([0, np.max(histogram) * 1.1])
    plt.xlabel('Brightness Value')
    plt.ylabel('Frequency')
    plt.show()