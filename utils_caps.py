import os
import scipy
import numpy as np
import tensorflow as tf
import glob

def load_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True, one_hot=False):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)

'''
def get_batch_data(dataset, batch_size, num_threads):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)
    elif dataset == 'fashion-mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)
'''


def get_batch_data(dataset_dir, caps_dataset_dir, batch_size, seed):
    # Load all capsdata
    file_list = sorted(glob.glob(os.path.join(caps_dataset_dir,'*.csv')))
    data_list = []
    for f in file_list:
        data = np.loadtxt(fname=f, delimiter=',')
        data_list.append(data)
    
    # Load the list of training files into queues
    text_path = os.path.join(dataset_dir, 'train.txt')
    with open(text_path, 'r') as f:
        frames = f.readlines()
    subfolders = [int(x.split(' ')[0]) for x in frames]
    frames_ids = [int(x.split(' ')[1][:-1]) for x in frames]
    #for i in range(10):
    #    temp = data_list[subfolders[i]][frames_ids[i]-1][0]
    #    print(temp)
    class_list = [int(data_list[subfolders[i]][frames_ids[i]-1][1]) for i in range(len(frames))]
    yaw_list = [data_list[subfolders[i]][frames_ids[i]-1][2] for i in range(len(frames))]
    tx_list = [data_list[subfolders[i]][frames_ids[i]-1][3] for i in range(len(frames))]
    ty_list = [data_list[subfolders[i]][frames_ids[i]-1][4] for i in range(len(frames))]
    tz_list = [data_list[subfolders[i]][frames_ids[i]-1][5] for i in range(len(frames))]
    #input_list = [data_list[subfolders[i]][frames_ids[i]-1][2:] for i in range(len(frames))]
    #print(np.shape(input_list))
    #exit()
    x_list = tf.stack([yaw_list, tx_list, ty_list, tz_list], axis=-1)
    data_queues = tf.train.slice_input_producer([x_list, class_list], seed=seed, shuffle=True)
    X, Y = tf.train.batch(data_queues, num_threads=8, 
            batch_size=batch_size,
            capacity=batch_size * 64,
            allow_smaller_final_batch=False)

    #(4,4) , (4,)
    return (X, Y)
            

def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)

if __name__ == "__main__":
    get_batch_data("/home/jwlim/hdd2/formatted_odom/", "/home/jwlim/hdd2/capsnet_data/", batch_size=10)
