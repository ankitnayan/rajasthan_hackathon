from django.shortcuts import render
from django.http import HttpResponse

import argparse
import tensorflow as tf

import imageio
import numpy as np
import scipy.io as sio
from PIL import Image

from chexnet.densenet import DenseNet

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.transform import resize

cmap = plt.get_cmap('jet')


args = {}

args['bc_mode'] = True
args['keep_prob'] = 1.0
args['predict'] = True
args['model_type'] = "DenseNet-BC"
args['growth_rate'] = 24
args['depth'] = 100
args['total_blocks'] = 4
args['reduction'] = 0.5

args['weight_decay'] = 1e-4
args['nesterov_momentum'] = 0.9
args['dataset'] = 'CHEXN'
args['should_save_logs'] = True
args['should_save_model'] = True

class DataProvider:
    @property
    def data_shape(self):
        """Return shape as python list of one data entry"""
        raise NotImplementedError

    @property
    def n_classes(self):
        """Return `int` of num classes"""
        raise NotImplementedError

    def labels_to_one_hot(self, labels):
        """Convert 1D array of labels to one hot representation
        
        Args:
            labels: 1D numpy array
        """
        new_labels = np.zeros((labels.shape[0], self.n_classes))
        print("labels.shape",labels.shape)
        print("labels", labels)
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels

    def labels_from_one_hot(self, labels):
        """Convert 2D array of labels to 1D class based representation
        
        Args:
            labels: 2D numpy array
        """
        return np.argmax(labels, axis=1)


class DummyDataProvider(DataProvider):
    def __init__(self, save_path=None, validation_set=None,
                 validation_split=None, shuffle=False, normalization=None,
                 one_hot=True, **kwargs):
        pass
        
    @property
    def n_classes(self):
        return 6

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join(tempfile.gettempdir(), 'chexn')
        return self._save_path

    @property
    def data_url(self):
        return "http://ufldl.stanford.edu/housenumbers/"

    @property
    def data_shape(self):
        return (224,224, 1)



model_params = args


data_provider = DummyDataProvider()

print("Initialize the model..")
model = DenseNet(data_provider=data_provider, **model_params)


model.load_model(model_path='./chexnet/save_60k/model.chkpt-39')

height = 224
width = 224



def index(request):


	image_path = request.GET.get('image_path', None)
	print (image_path)

	if not model_params['predict']:
		return HttpResponse("Hello, world. You're predicting from model")

	
	im = Image.open(image_path)
	im = im.resize((width,height), Image.BILINEAR)
	im = np.asarray(im)
	im = im/255

	if (len(im.shape) > 2):
	    im = im[:,:,0]

	#img_3D = np.dstack((im,im,im))
	img_4D = im.reshape((-1, width, height, 1))
	print ("shape of input image: ", img_4D.shape)


	last_block_output, classes_softmax, prediction_class = model.predict(img_4D)
	print (prediction_class)

	'''
	sess = tf.Session()

	new_saver = tf.train.import_meta_graph('./chexnet/save_60k/model.chkpt-39.meta')

	new_saver.restore(sess,tf.train.latest_checkpoint('./chexnet/save_60k/'))

	graph = tf.get_default_graph()
	op2restore = graph.get_tensor_by_name('Transition_to_classes/W/read:0')

	weights = sess.run(op2restore)
	print ("shape of weights: ", weights.shape)
	heatmap = np.matmul(last_block_output, weights)[:,:,prediction_class[0]]

	sess.close()

	heatmap = ((heatmap - heatmap.min())/(heatmap.max()-heatmap.min())*255).astype('uint8')

	heatmap_img = Image.fromarray(heatmap)
	heatmap_img.save('./chexnet/heatmap.jpg')

	print ("Heatmap image made !!!")
	'''
	return HttpResponse(str(classes_softmax)+"<br>"+str(prediction_class))
