import os
import time
import numpy as np
import tensorflow as tf
from scipy import ndimage, misc
from loader import Loader

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('image', '', 'Image File')
flags.DEFINE_string('model', '', 'Model File.')
flags.DEFINE_string('type', 'ResNet', 'ResNet/SSAI')

def data_processor(filename, config):
	input_image = ndimage.imread('{0}'.format(filename))
	input_image = (input_image - np.mean(input_image)) / np.std(input_image)
	return (filename, input_image)

if FLAGS.model == 'ResNet':
	import model_residual as m
else:
	import model_ssai as m

if os.path.isdir("segmentations") == False:
	os.mkdir("segmentations")

saved_model = FLAGS.model
filename = FLAGS.image
ext = os.path.splitext(os.path.basename(filename[0]))[1]

if ext not in ['.jpg', '.png', '.tiff', '.tif']:
	print('The file you have chosen is not jpg, png, tiff, or tiff!')
else:
	files = [filename]
	data_loader = Loader(files, 1, processor=data_processor, randomize=True, augment=False)
	data_loader.start()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = False
	config.allow_soft_placement = False
	config.log_device_placement = False

	with tf.Session(config=config) as sess:
		saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
		saver.restore(sess, saved_model)

		batch = data_loader.get_batch(1)
		filenames = batch[0]
		input_images = batch[1]

		result = sess.run([m.result], feed_dict={ 
			m.input_image: input_images,
			m.is_train: False
		})

		self.filename = os.path.splitext(os.path.basename(filename))[0] + '_' + type + '_' + model + '.jpg'
		result = result[0].reshape((1500, 1500))
		misc.imsave("segmentations/{0}".format(self.filename), result)

	data_loader.stop()
