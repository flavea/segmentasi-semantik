from os import listdir
import os
from random import shuffle, randint
import time

import numpy as np
import tensorflow as tf
from scipy import ndimage, misc

from loader import Loader

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'ResNet', 'ResNet/SSAI')
flags.DEFINE_integer('batch_size', 1, 'Batch Size.')
flags.DEFINE_string('type', 'building', 'Building/Road')
flags.DEFINE_integer('training_step', 5000, 'Training Step')
flags.DEFINE_integer('saved', 100, 'Save training result every x step')

def data_processor(filename, config):
	if FLAGS.type == 'building':
		input_image = ndimage.imread('building_input_images/{0}'.format(filename))
		label_image = ndimage.imread('building_label_images/{0}'.format(filename[:-1]))
	else:
		input_image = ndimage.imread('road_input_images/{0}'.format(filename))
		label_image = ndimage.imread('road_label_images/{0}'.format(filename[:-1]))

	label_image = label_image[:, :, :1] / 255

	if config.rotate:
		angle = randint(0, 360)
		input_image = ndimage.rotate(input_image, angle, reshape=False)
		label_image = ndimage.rotate(label_image, angle, reshape=False)

	input_image = (input_image - np.mean(input_image)) / np.std(input_image)
	return (filename, input_image, label_image)

if FLAGS.model == 'ResNet':
	import model_residual as m
else:
	import model_ssai as m

modelName = m.modelName
type = FLAGS.type

if FLAGS.type == 'building':
	label_files = [filename for filename in listdir('building_input_images')]
else:
	label_files = [filename for filename in listdir('road_input_images')]

if os.path.isdir("test-results-{0}-{1}".format(type, modelName)) == False:
	os.mkdir("test-results-{0}-{1}".format(type, modelName))

label_files.sort()

test_label_files = label_files[:5]
test = Loader(test_label_files, 5, processor=data_processor)
test.start()
test_batch = test.get_batch(1)
test.stop()

batch_size = FLAGS.batch_size
train_label_files = label_files[batch_size:]
train = Loader(train_label_files, batch_size * 5, processor=data_processor, randomize=True, rotate=True)
train.start()

config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.allow_soft_placement = False
config.log_device_placement = False

with tf.Session(config=config) as sess:
	saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
	summary_writer = tf.summary.FileWriter('models/{0}-{1}'.format(modelName, type), graph=sess.graph)
	summary_writer_test = tf.summary.FileWriter('models/{0}-{1}-test'.format(modelName, type), graph=sess.graph)

	sess.run(tf.global_variables_initializer())
	
	print("Training start!")

	while True:
		batch = train.get_batch(batch_size)
		filenames = batch[0]
		input_images = batch[1]
		label_images = batch[2]

		_, error, summary, step = sess.run([m.train, m.error, m.summary, m.global_step], feed_dict={ 
			m.input_image: input_images,	
			m.label_image: label_images
		})

		summary_writer.add_summary(summary, step)

		if step % (FLAGS.saved/batch_size) == 0:
			print("{0}: step {1}; error {2};".format(time.time() - startTime, step, error))
			path = 'models/model' + '-' + str(FLAGS.type) + '-' + modelName + '/' + str(step) + '.ckpt'
			print("Saving model to " + path)
			save_path = saver.save(sess, path)
			
			filenames = test_batch[0]
			input_images = test_batch[1]
			label_images = test_batch[2]

			test_error, learning_rate, result, summary = sess.run([m.error, m.learning_rate, m.result, m.test_summary], feed_dict={ 
				m.input_image: input_images,	
				m.label_image: label_images,
				m.is_train: False
			})

			summary_writer_test.add_summary(summary, step)

			print("Test start!")
			print("{0}: step {1}; error {2};".format(time.time() - startTime, step, error))

			filename = filenames[0]
			result_image = result[0].reshape((1500, 1500))
			misc.imsave("test-results-{0}-{1}/{2}-{3}.png".format(type, modelName, filename, step), result_image)

		if step == FLAGS.training_step: break

train.stop()
