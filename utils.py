import skimage
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf

# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))

# All histogram_matching functions are from:
# https://github.com/rejunity/unfair-advantage
def searchsortedN(sorted_array, values_to_search, n):
    from tensorflow.python.ops import array_ops
    
    indices = array_ops.zeros_like(values_to_search, dtype=tf.float32)
    n = int(n)
    
    while n > 1:
        n = n / 2

        idxL = indices
        idxR = indices + tf.to_float(n)

        pred = tf.less(values_to_search, tf.gather(sorted_array, tf.to_int32(idxR)))
        indices = tf.where(pred, idxL, idxR)

    pred = tf.less(values_to_search, sorted_array[0])
    indices = tf.where(pred, indices, indices + 1)
    return indices

def interp_linear(x_new, x, y, nbins):
    from tensorflow.python.framework import dtypes
    from tensorflow.python.ops import clip_ops
    from tensorflow.python.ops import math_ops

    x_new_indices = searchsortedN(x, x_new, nbins)

    lo = x_new_indices - 1
    hi = x_new_indices

    # Clip indices so that they are within the range
    hi = math_ops.cast(
        clip_ops.clip_by_value(hi, 0, nbins-1), dtypes.int32)
    lo = math_ops.cast(
        clip_ops.clip_by_value(lo, 0, nbins-1), dtypes.int32)

    x_lo = tf.gather(x, lo) #x_lo = x[lo]
    x_hi = tf.gather(x, hi) #x_hi = x[hi]
    y_lo = tf.gather(y, lo) #y_lo = y[lo]
    y_hi = tf.gather(y, hi) #y_hi = y[hi]

    # Calculate the slope of regions that each x_new value falls in.
    dx = (x_hi - x_lo)
    slope = (y_hi - y_lo) / dx

    # Calculate the actual value for each entry in x_new.
    y_linear = slope*(x_new - x_lo) + y_lo
    y_nearest = y_lo

    # Protect against NaN (div-by-zero)
    p = tf.not_equal(dx, 0.0)
    y_new = tf.where(p, y_linear, y_nearest)

    return y_new

def histogram_fixed_width(values, value_range, nbins=100):
    from tensorflow.python.framework import dtypes
    from tensorflow.python.ops import clip_ops
    from tensorflow.python.ops import math_ops
    from tensorflow.python.ops import array_ops

    nbins_float = math_ops.to_float(nbins)

    # Map tensor values that fall within value_range to [0, 1].
    scaled_values = math_ops.truediv(values - value_range[0],
                                     value_range[1] - value_range[0],
                                     name='scaled_values')

    # map tensor values within the open interval value_range to {0,.., nbins-1},
    # values outside the open interval will be zero or less, or nbins or more.
    indices = math_ops.floor(nbins_float * scaled_values, name='indices')

    # Clip edge cases (e.g. value = value_range[1]) or "outliers."
    indices = math_ops.cast(
        clip_ops.clip_by_value(indices, 0, nbins_float - 1), dtypes.int32)

    #counts = tf.Variable(...) <= array_ops.zeros_like(indices, dtype=dtypes.int32))
    #return tf.scatter_add(counts, indices, array_ops.ones_like(indices, dtype=dtypes.int32)), indices
    print('test')
    print(indices, nbins)
    return math_ops.unsorted_segment_sum(
        array_ops.ones_like(indices, dtype=dtypes.float32),
        indices,
        nbins), indices

def feature_histogram_matching(source, template, value_range, nbins):

    s_value_range = [tf.reduce_min(source), tf.reduce_max(source)]
    t_value_range = [tf.reduce_min(template), tf.reduce_max(template)]
    #s_value_range = value_range
    #t_value_range = value_range
    t_values = tf.linspace(t_value_range[0], t_value_range[1], nbins)

    s_counts, indices = histogram_fixed_width(source, s_value_range, nbins)
    t_counts, _ = histogram_fixed_width(template, t_value_range, nbins)

    s_cdf = tf.to_float(s_counts)
    s_cdf = tf.cumsum(s_cdf)
    s_cdf /= s_cdf[-1]

    t_cdf = tf.to_float(t_counts)
    t_cdf = tf.cumsum(t_cdf)
    t_cdf /= t_cdf[-1]

    interp_t_values = interp_linear(s_cdf, t_cdf, t_values, nbins)
    interp_t_values = tf.maximum(interp_t_values, 0.0)
    values = tf.gather(interp_t_values, indices)
    return values

def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
