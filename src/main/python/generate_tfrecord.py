import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import PIL
import torchvision.transforms as T
import pandas as pd
from tqdm import tqdm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



SIZE = 256
transforms = [
    T.Resize(SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
]

transforms = T.Compose(transforms)



flickr = pd.read_csv('data/count/flickr30k_train_captions_counts.csv')
flickr['path'] = '/path/to/flickr30k/' + flickr['id'].astype(str) + '.jpg'

openimg = pd.read_csv('data/count/open_images_train_v6_captions_counts.csv')
openimg['path'] = '/path/toy/Open Images/train/' + openimg['id'].astype(str) + '.jpg'

coco = pd.read_csv('data/count/coco_train_captions_counts.csv')
coco['path'] = ['/path/to/COCO/train2017/' + s.zfill(12) + '.jpg' for s in coco['id'].astype(str)]

ade = pd.read_csv('data/count/ade20k_train_captions_counts.csv')
ade['path'] = '/path/to/ADE20K/images/training/' + ade['id'].astype(str) + '.jpg'

stanford = pd.read_csv('data/count/stanford_paragraphs_counts.csv')
stanford['path'] = '/path/to/stanford_paragraphs/images/' + stanford['id'].astype(str) + '.jpg'

train_df = pd.concat([flickr, openimg, coco, ade, stanford]).reset_index(drop=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)


flickr = pd.read_csv('data/count/flickr30k_test_captions_counts.csv')
flickr['path'] = '/path/to/flickr30k/' + flickr['id'].astype(str) + '.jpg'

openimg = pd.read_csv('data/count/open_images_test_captions_counts.csv')
openimg['path'] = '/path/to/Open Images/test/' + openimg['id'].astype(str) + '.jpg'

coco = pd.read_csv('data/count/coco_val_captions_counts.csv')
coco['path'] = ['/path/to/COCO/val2017/' + s.zfill(12) + '.jpg' for s in coco['id'].astype(str)]

ade = pd.read_csv('data/count/ade20k_validation_captions_counts.csv')
ade['path'] = '/path/to/ADE20K/images/validation/' + ade['id'].astype(str) + '.jpg'

test_df = pd.concat([flickr, openimg, coco, ade]).reset_index(drop=True)


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(img, count):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature = {
        'img': _bytes_feature(tf.io.serialize_tensor(img)),
        'count': _int64_feature(count)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


writer = tf.io.TFRecordWriter('data/tfrecord/train.tfrecord')


for index, row in tqdm(train_df.iterrows(), total = len(train_df)):
    try:
        img = transforms(PIL.Image.open(row["path"]).convert('RGB'))
        count = row["count"]
        writer.write(serialize_example(img.numpy(), count))
    except IOError:
        print('error loading image', row["path"])

writer.close()


writer = tf.io.TFRecordWriter('data/tfrecord/test.tfrecord')


for index, row in tqdm(test_df.iterrows(), total = len(train_df)):
    try:
        img = transforms(PIL.Image.open(row["path"]).convert('RGB'))
        count = row["count"]
        writer.write(serialize_example(img.numpy(), count))
    except IOError:
        print('error loading image', row["path"])

writer.close()