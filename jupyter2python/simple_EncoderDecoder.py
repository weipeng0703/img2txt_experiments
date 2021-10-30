# Linux下使用plt
import matplotlib
matplotlib.use('Agg')

# from google.colab import drive
# drive.mount('/content/drive')
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib #for saving model files as pkl files
import os
import seaborn as sns
import cv2
import imgaug.augmenters as iaa
sns.set(palette='muted',style='white')
import tensorflow as tf
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, Input, Embedding, LSTM,Dot,Reshape,Concatenate,BatchNormalization, GlobalMaxPooling2D, Dropout, Add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu #bleu score
tf.compat.v1.enable_eager_execution()
import os
import math
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="false" #https://github.com/tensorflow/tensorflow/issues/33721#issuecomment-577307175



image_folder = '/home/vpeng/datasets/chestX/NLMCXR_png/NLMCXR_png'
df_path = '/home/vpeng/img2txt/jupyter2python/df_final.pkl'
chexnet_weights = '/home/vpeng/img2txt/jupyter2python/chexnet_weights/brucechou1983_CheXNet_Keras_0.3.0_weights.h5'


# create tokenizer

df = pd.read_pickle(df_path)
col = ['image_1','image_2','impression','xml file name']
df = df[col].copy()
#path
df['image_1'] = df['image_1'].apply(lambda row: os.path.join(image_folder,row)) #https://stackoverflow.com/a/61880790
df['image_2'] = df['image_2'].apply(lambda row: os.path.join(image_folder,row))

df['impression_final'] = '<CLS> ' + df.impression + ' <END>'
df['impression_ip'] = '<CLS> ' + df.impression
df['impression_op'] = df.impression + ' <END>'
print(df.shape)
print(df.head(2))

# 查看相同的caption出现了多少次

print(df['impression'].value_counts())

# upsample all those datapoint which impression value counts <=5

# remove all those datapoints which are duplicated based on xml file name
# (ie of the same patient because some patients are having more than 2 images)
# and then split the data into train and test.

df.drop_duplicates(subset = ['xml file name'], inplace = True)

# adding a new column impression counts which tells the total value counts of
# impression of that datapoint
k = df['impression'].value_counts()
df = df.merge(k,
         left_on = 'impression',
         right_index=True) #join left impression value with right index

# print(df.columns)

df.columns = ['impression', 'image_1', 'image_2', 'impression_x', 'xml file name','impression_final',
       'impression_ip', 'impression_op', 'impression_counts'] #changin column names
del df['impression_x'] #deleting impression_x column
# print(df.head())

# 1.divide the data into two one
# with all the impression value counts greater than 5 (other1)
# and other being <=5 (other2). Then split the data with test_size=0.1.
# 2.A sample of 0.05*other2.shape[0] will be then taken
# and will be added on to the test data that was split.
# 3.The other data from the other2 will be appended to train.

other1 = df[df['impression_counts']>5] #selecting those datapoints which have impression valuecounts >5
other2 = df[df['impression_counts']<=5] #selecting those datapoints which have impression valuecounts <=5
train,test = train_test_split(other1,stratify = other1['impression'].values,test_size = 0.1,random_state = 420)
test_other2_sample = other2.sample(int(0.2*other2.shape[0]),random_state = 420) #getting some datapoints from other2 data for test data
other2 = other2.drop(test_other2_sample.index,axis=0)
#here i will be choosing 0.5 as the test size as to create a reasonable size of test data
test = test.append(test_other2_sample)
test = test.reset_index(drop=True)

train = train.append(other2)
train = train.reset_index(drop=True)
print(train.shape[0])
print(test.shape[0])

# upsample and downsample certain datapooints.

df_majority = train[train['impression_counts']>=100] #having value counts >=100
df_minority = train[train['impression_counts']<=5] #having value counts <=5
df_other = train[(train['impression_counts']>5)&(train['impression_counts']<100)] #value counts between 5 and 100
n1 = df_minority.shape[0]
n2 = df_majority.shape[0]
n3 = df_other.shape[0]
#we will upsample them to 30
df_minority_upsampled = resample(df_minority,
                                 replace = True,
                                 n_samples = 3*n1,
                                 random_state = 420)
df_majority_downsampled = resample(df_majority,
                                 replace = False,
                                 n_samples = n2//15,
                                 random_state = 420)
df_other_downsampled = resample(df_other,
                                 replace = False,
                                 n_samples = n3//10,
                                 random_state = 420)

train = pd.concat([df_majority_downsampled ,df_minority_upsampled,df_other_downsampled])
train = train.reset_index(drop=True)
# del df_minority_upsampled,df_minority,df_majority,df_other,df_other_downsampled
print(train.shape)

print(train.impression.value_counts())

# 生成train.pkl和test.pkl

# folder_name = '/home/vpeng/img2txt/jupyter2python'
# file_name = 'train.pkl'
# train.to_pickle(os.path.join(folder_name,file_name))
# print("已生成 train.pkl")
#
# file_name = 'test.pkl'
# test.to_pickle(os.path.join(folder_name,file_name))
# print("已生成 test.pkl")


folder_name = '/home/vpeng/img2txt/jupyter2python'
file_name = 'train.pkl'
train = pd.read_pickle(os.path.join(folder_name,file_name))

file_name = 'test.pkl'
test = pd.read_pickle(os.path.join(folder_name,file_name))

#tokenizer
tokenizer = Tokenizer(filters = '',oov_token = '<unk>') #setting filters to none
tokenizer.fit_on_texts(train.impression_final.values)
train_captions = tokenizer.texts_to_sequences(train.impression_final)
test_captions = tokenizer.texts_to_sequences(test.impression_final)
vocab_size = len(tokenizer.word_index)
caption_len = np.array([len(i) for i in train_captions])
start_index = tokenizer.word_index['<cls>'] #tokened value of <cls>
end_index = tokenizer.word_index['<end>'] #tokened value of <end>


#visualising impression length and other details
ax = sns.displot(caption_len,height = 6)
ax.set_titles('Value Counts vs Caption Length')
ax.set_xlabels('Impresion length')
plt.show()
# plt.savefig("Value Counts for caption length top 5 values.png")
print('\nValue Counts for caption length top 5 values\n')
print('Length|Counts')
print(pd.Series(caption_len).value_counts()[:5])
print('\nThe max and min value of "caption length" was found to be %i and %i respectively'%(max(caption_len),min(caption_len)))
print('The 80 percentile value of caption_len which is %i will be taken as the maximum padded value for each impression'%(np.percentile(caption_len,80)))
max_pad = int(np.percentile(caption_len,80))

del train_captions,test_captions #we will create tokenizing  and padding in-built in dataloader
