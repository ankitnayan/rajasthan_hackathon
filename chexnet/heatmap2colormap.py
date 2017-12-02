import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import matplotlib.image as mpimg

cmap = plt.get_cmap('jet')

from skimage.transform import resize


for i in range(0,10):
	heatmap = mpimg.imread('./heatmap.png')

	heatmap1 = resize(heatmap, (224, 224))
	print ("heatmap_shape: ", heatmap1.shape)
	rgba_img = cmap(heatmap1)
	plt.imshow(rgba_img, cmap='jet')
	#plt.colorbar()
	#plt.show()
	#plt.savefig('./color_heatmap.png') 

	large_img = mpimg.imread('./data/pneumothorax/00000416_005.png')
	large_img1 = resize(large_img, (224, 224))
	print ("image_shape: ", large_img1.shape)
	plt.imshow(large_img1, alpha=0.6, cmap='gray')
	plt.savefig(str(i)+'colormap.png')
	#plt.clf()