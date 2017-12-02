import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import matplotlib.image as mpimg

cmap = plt.get_cmap('jet')
heatmap = mpimg.imread('./heatmap.png')
from skimage.transform import resize
heatmap1 = resize(heatmap, (224, 224))
rgba_img = cmap(heatmap1)
plt.imshow(rgba_img, cmap='jet')
plt.colorbar()
#plt.show()
#plt.savefig('./color_heatmap.png') 

large_img = mpimg.imread('./data/pneumothorax/00000416_005.png')
large_img = resize(large_img, (224, 224))
plt.imshow(large_img, alpha=0.6, cmap='gray')
plt.savefig('colormap.jpg')
