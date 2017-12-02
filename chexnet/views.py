from django.shortcuts import render
from django.http import HttpResponse

import argparse
import tensorflow as tf

import imageio
import numpy as np
import scipy.io as sio
from PIL import Image

from models.dense_net import DenseNet
from data_providers.utils import get_data_provider_by_name

train_params_cifar = {
    'batch_size': 64,
    'n_epochs': 300,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,  # epochs * 0.5
    'reduce_lr_epoch_2': 225,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}

train_params_svhn = {
    'batch_size': 64,
    'n_epochs': 40,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20,
    'reduce_lr_epoch_2': 30,
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': True,  # shuffle dataset every epoch or not
    'normalization': 'divide_255',
}

train_params_chexn = {
    'batch_size': 16,
    'n_epochs': 40,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20,
    'reduce_lr_epoch_2': 30,
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': True,  # shuffle dataset every epoch or not
    'normalization': 'divide_255',
}

def get_train_params_by_name(name):
    if name in ['C10', 'C10+', 'C100', 'C100+']:
        return train_params_cifar
    if name == 'SVHN':
        return train_params_svhn
    if name == 'CHEXN':
        return train_params_chexn

def index(request):

	parser = argparse.ArgumentParser()

	parser.add_argument(
	    '--train', action='store_true',
	    help='Train the model')

	parser.add_argument(
	    '--predict', action='store_true',
	    help='Predict from model')

	parser.add_argument(
	    '--test', action='store_true',
	    help='Test model for required dataset if pretrained model exists.'
	         'If provided together with `--train` flag testing will be'
	         'performed right after training.')
	parser.add_argument(
	    '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'],
	    default='DenseNet',
	    help='What type of model to use')
	parser.add_argument(
	    '--growth_rate', '-k', type=int, choices=[12, 24, 40],
	    default=12,
	    help='Grows rate for every layer, '
	         'choices were restricted to used in paper')
	parser.add_argument(
	    '--depth', '-d', type=int, choices=[40, 100, 190, 250],
	    default=40,
	    help='Depth of whole network, restricted to paper choices')
	parser.add_argument(
	    '--dataset', '-ds', type=str,
	    choices=['C10', 'C10+', 'C100', 'C100+', 'SVHN','CHEXN'],
	    default='CHEXN',
	    help='What dataset should be used')
	parser.add_argument(
	    '--total_blocks', '-tb', type=int, default=3, metavar='',
	    help='Total blocks of layers stack (default: %(default)s)')
	parser.add_argument(
	    '--keep_prob', '-kp', type=float, metavar='',
	    help="Keep probability for dropout.")
	parser.add_argument(
	    '--weight_decay', '-wd', type=float, default=1e-4, metavar='',
	    help='Weight decay for optimizer (default: %(default)s)')
	parser.add_argument(
	    '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
	    help='Nesterov momentum (default: %(default)s)')
	parser.add_argument(
	    '--reduction', '-red', type=float, default=0.5, metavar='',
	    help='reduction Theta at transition layer for DenseNets-BC models')

	parser.add_argument(
	    '--logs', dest='should_save_logs', action='store_true',
	    help='Write tensorflow logs')
	parser.add_argument(
	    '--no-logs', dest='should_save_logs', action='store_false',
	    help='Do not write tensorflow logs')
	parser.set_defaults(should_save_logs=True)

	parser.add_argument(
	    '--saves', dest='should_save_model', action='store_true',
	    help='Save model during training')
	parser.add_argument(
	    '--no-saves', dest='should_save_model', action='store_false',
	    help='Do not save model during training')
	parser.set_defaults(should_save_model=True)

	parser.add_argument(
	    '--renew-logs', dest='renew_logs', action='store_true',
	    help='Erase previous logs for model if exists.')
	parser.add_argument(
	    '--not-renew-logs', dest='renew_logs', action='store_false',
	    help='Do not erase previous logs for model if exists.')
	parser.set_defaults(renew_logs=True)

	args = parser.parse_args()

	print type(args)

	if not args.keep_prob:
	    if args.dataset in ['C10', 'C100', 'SVHN','CHEXN']:
	        args.keep_prob = 0.8
	    else:
	        args.keep_prob = 1.0
	if args.model_type == 'DenseNet':
	    args.bc_mode = False
	    args.reduction = 1.0
	elif args.model_type == 'DenseNet-BC':
	    args.bc_mode = True

	model_params = vars(args)

	if not args.train and not args.test and not args.predict:
	    print("You should train or test or predict your network. Please check params.")
	    exit()

	# some default params dataset/architecture related
	train_params = get_train_params_by_name(args.dataset)
	print("Params:")
	for k, v in model_params.items():
	    print("\t%s: %s" % (k, v))
	print("Train params:")
	for k, v in train_params.items():
	    print("\t%s: %s" % (k, v))


	if args.predict:
	    data_provider = DummyDataProvider()

	else:
	    data_provider = get_data_provider_by_name(args.dataset, train_params)


	print("Initialize the model..")
	model = DenseNet(data_provider=data_provider, **model_params)

	'''
	if args.train:
	    print("Data provider train images: ", data_provider.train.num_examples)
	    model.train_all_epochs(train_params)


	if args.test:
	    if not args.train:
	        model.load_model()
	    print("Data provider test images: ", data_provider.test.num_examples)
	    print("Testing...")
	    loss, accuracy = model.test(data_provider.test, batch_size=200)
	    print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))


	if args.predict:

	    model.load_model(model_path='./chexnet/save_60k/model.chkpt')


	    height = 224
	    width = 224

	    filename = './chexnet/data/pneumonia/99.png'
	    im = Image.open(filename)
	    im = im.resize((width,height), Image.BILINEAR)
	    im = np.asarray(im)
	    im = im/255

	    if (len(im.shape) > 2):
	        im = im[:,:,0]

	    #img_3D = np.dstack((im,im,im))
	    img_4D = img_3D.reshape((-1, width, height, 1))
	    print ("shape of input image: ", img_4D.shape())


	    last_block_output, prediction_class = model.predict(img_4D)
	    print (prediction_class)

	    sess = tf.Session()

	    new_saver = tf.train.import_meta_graph('/home/ankit/Desktop/vision_networks/saves/DenseNet-BC_growth_rate=12_depth=40_dataset_CHEXN/model.chkpt.meta')

	    new_saver.restore(sess,tf.train.latest_checkpoint('./saves/DenseNet-BC_growth_rate=12_depth=40_dataset_CHEXN/'))

	    graph = tf.get_default_graph()
	    op2restore = graph.get_tensor_by_name('Transition_to_classes/W/read:0')

	    weights = sess.run(op2restore)
	    heatmap = np.matmul(last_block_output, weights)[:,:,prediction_class[0]]
	    
	    sess.close()
	'''

	return HttpResponse("Hello, world. You're at the polls index.")
