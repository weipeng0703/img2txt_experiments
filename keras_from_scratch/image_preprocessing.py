# 使用预先训练的模型获取 "image feature"并将其保存到文件中。
# 可以在以后加载这些特征，并将它们作为数据集中某张照片的解释输入我们的模型。
# 将数据集中的图片转换为.pkl格式的图像特征的字典。

from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	model = VGG16()
	# re-structure the model
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# summarize
	print(model.summary())

	# extract features from each photo
	features = dict()

	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
        # print("reshape前:")
        # print(image.shape)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # print("reshape后：")
        # print(image.shape)
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features

if __name__ == '__main__':
	# extract features from all images
	directory = '/home/vpeng/datasets/flickr8k/images'
	features = extract_features(directory)
	print('Extracted Features(sum of images): %d' % len(features))
	# save to file
	dump(features, open('features.pkl', 'wb'))