import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = Image.open("datasets/v1/insufficient/1766269046.pfm")
img = np.array(img) / 2500

plt.figure()
plt.imshow(img)
plt.show()