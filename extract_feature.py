from collections import OrderedDict
import time

import numpy as np
import scipy.io as sio
import os
import sys
import caffe


def load_image_list(img_dir, list_file_name):
    list_file_path = os.path.join(img_dir, list_file_name)
    f = open(list_file_path, 'r')
    image_fullpath_list = []
    image_list = []
    labels = []
    for line in f:
        items = line.split()
        image_list.append(items[0].strip())
        image_fullpath_list.append(os.path.join(img_dir, items[0].strip()))
        labels.append(items[1].strip())
    return image_fullpath_list, labels, image_list


def blobs_data(blob):
    try:
        d = blob.const_data
        #print 'GPU mode.'
    except AttributeError:
        #print 'GPU mode not support.'
        d = blob.data
    return d


def blobs_diff(blob):
    try:
        d = blob.const_diff
    except AttributeError:
        #print 'GPU mode not support.'
        d = blob.diff
    return d


def detect_GPU_extract_support(net):
    k, blob = net.blobs.items()[0]
    gpu_support = 0
    try:
        d = blob.const_data
        gpu_support = 1
    except AttributeError:
        gpu_support = 0
    return gpu_support


def extract_feature(network_proto_path,
                    network_model_path,
                    image_list, data_mean, layer_name, image_as_grey = False):
    """
    Extracts features for given model and image list.

    Input
    network_proto_path: network definition file, in prototxt format.
    network_model_path: trainded network model file
    image_list: A list contains paths of all images, which will be fed into the
                network and their features would be saved.
    layer_name: The name of layer whose output would be extracted.
    save_path: The file path of extracted features to be saved.
    """
    net = caffe.Net(network_proto_path, network_model_path, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_input_scale('data', 1)
    transformer.set_transpose('data', (2, 0, 1))
    blobs = OrderedDict([(k, v.data) for k, v in net.blobs.items()])

    shp = blobs[layer_name].shape
    print blobs['data'].shape

    batch_size = blobs['data'].shape[0]
    print blobs[layer_name].shape

    features_shape = (len(image_list), shp[1])
    features = np.empty(features_shape, dtype='float32', order='C')
    for idx, path in zip(range(features_shape[0]), image_list):
        img = caffe.io.load_image(path, color=False)
        prob = net.forward_all(data=np.asarray([transformer.preprocess('data', img)]))
        print np.shape(prob['prob'])
        blobs = OrderedDict([(k, v.data) for k, v in net.blobs.items()])
        features[idx, :] = blobs[layer_name][0, :].copy()
        print '%d images processed' % (idx + 1)
    features = np.asarray(features, dtype='float32')
    return features


def extract_features_to_mat(network_proto_path, network_model_path, data_mean,
                            image_dir,  list_file, layer_name, save_path, image_as_grey = False):
    img_list, labels, img_list_original = load_image_list(image_dir, list_file)
    print img_list[0:10]
    print labels[0:10]

    ftr = extract_feature(network_proto_path, network_model_path,
                          img_list, data_mean, layer_name, image_as_grey)

    float_labels = labels_list_to_float(labels)
    dic = {'features': ftr,
           'labels': float_labels,
           'labels_original': string_list_to_cells(labels),
           'image_path': string_list_to_cells(img_list_original)}
    sio.savemat(save_path, dic)
    return


def string_list_to_cells(lst):
    """
    Uses numpy.ndarray with dtype=object. When save to mat file using scipy.io.savemat, it will be a cell array.
    """
    cells = np.ndarray(len(lst), dtype = 'object')
    for i in range(len(lst)):
        cells[i] = lst[i]
    return cells


def labels_list_to_float(labels):

    int_labels = []
    for e in labels:
        try:
            inte = int(e)
        except ValueError:
            print 'Labels are not int numbers. A mapping will be used.'
            break
        int_labels.append(inte)
    if len(int_labels) == len(labels):
        return int_labels

    labels_unique = list(sorted(set(labels)))
    print labels[0:10]
    print labels_unique[0:10]

    dic = dict([(lb, i) for i, lb in zip(range(len(labels_unique)),labels_unique)])
    labels_float = [dic[a] for a in labels]
    return labels_float


def main(argv):

    #print argv[0]
    #print argv[0].lower()
    if len(argv) == 0:
        print 'To extract features:'
        print '  Extracts features and saves to mat file.'
        print '  Usage: python caffe_ftr.py network_def trained_model image_dir image_list_file layer_name save_file'
        print '    network_def: network definition prototxt file'
        print '    trained_model: trained network model file, such as deep_iter_10000'
        print '    image_dir: the root dir of images'
        print '    image_list_file: a txt file, each line contains an image file path relative to image_dir and a label, seperated by space'
        print '    layer_name: name of the layer, whose outputs will be extracted'
        print '    save_file: the file path to save features, better to ends with .mat'
        print 'To save filters:'
        print '  Saves filters to mat files.'
        print '  Usage: python caffe_ftr.py --save-filters network_def network_model save_path'
        print '    (args are similar.)'

        exit()

    cmd_str = argv[0].lower()

    if not cmd_str.startswith('--'):
        # old version
        if len(argv) != 6:
            print ' Extracts features and saves to mat file.'
            print ' Usage: python caffe_ftr.py network_def trained_model image_dir image_list_file layer_name save_file'
            print '    network_def: network definition prototxt file'
            print '    trained_model: trained network model file, such as deep_iter_10000'
            print '    image_dir: the root dir of images'
            print '    image_list_file: a txt file, each line contains an image file path relative to image_dir and a label, seperated by space'
            print '    layer_name: name of the layer, whose outputs will be extracted'
            print '    save_file: the file path to save features, better to ends with .mat'
            exit()
        start_time = time.time()
        extract_features_to_mat(argv[0], argv[1], None, argv[2], argv[3], argv[4], argv[5])
        end_time = time.time()
        print 'time used: %f s\n' % (end_time - start_time,)
        exit()

    return


if __name__ == '__main__':
    main(sys.argv[1:])