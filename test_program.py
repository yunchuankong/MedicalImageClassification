import tensorflow as tf
import numpy as np
from vgg_transfer_utils import load_scan,get_pixels_hu,normalize,triplicate,_3D_images_resize,vgg16

# test code here
slice_example = load_scan("example_dcm_sample/ANON_LUNG_TC001/CT")
slice_example_pixels = get_pixels_hu(slice_example)
normalized_example_pixels = normalize(slice_example_pixels)
tri_example_pixels = triplicate(normalized_example_pixels)
resize_example_pixels = _3D_images_resize(tri_example_pixels, 224,224)

sess = tf.Session()
imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
bottle_neck = sess.run(vgg.bn, feed_dict={vgg.imgs: resize_example_pixels})
