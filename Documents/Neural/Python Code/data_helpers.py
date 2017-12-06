import skimage
import selectivesearch
import numpy as np
import matplotlib.pyplot as plt
import re

img_width = 120
img_height = 120
labels = [
    'jpg_aeroplane'
    'jpg_bicycle',
    'jpg_bird',
    'jpg_boat',
    'jpg_bottle',
    'jpg_bus',
    'jpg_car',
    'jpg_cat',
    'jpg_chair',
    'jpg_cow',
    'jpg_diningtable',
    'jpg_dog',
    'jpg_horse',
    'jpg_motorbike',
    'jpg_person',
    'jpg_pottedplant',
    'jpg_sheep',
    'jpg_sofa',
    'jpg_train',
    'jpg_tvmonitor',
]


def get_onehots():
    length = len(labels)
    ret = []
    for i in range(length, -1, -1):
        ret.append([0] * (length - i) + [1] + [0] * i)
    return np.array(ret)


def get_train_data(img_name):
    plt.ion()
    # load some image from our data set
    img = skimage.io.imread('NC_data/train/{}.jpg'.format(img_name))
    # build up sequences to crop each region by, and
    f = open('NC_data/train/{}.txt'.format(img_name))
    crop_dict = {}
    for line in f.readlines():
        if not re.match(r'^.*,1\s0', line):
            parts = line.split(',')
            crop_dict[parts[0].split('.')[1]] = _build_seq(parts[1])
    region = _get_subimage(np.reshape(img, [400*400, 3]), list(crop_dict.values())[0])
    plt.imshow(skimage.transform.resize(region, (img_width, img_height)))
    plt.show(block=True)


def get_proposed_regions(img_name):
    ret = np.array()
    # load some image from our dataset
    img = skimage.io.imread('NC_data/test/{}'.format(img_name))
    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=400, sigma=0.8, min_size=400)
    candidates = set([r['rect'] for r in regions])
    for x, y, w, h in candidates:
        ret.put(skimage.transform.resize(img[x:x+h, y:y+h], (img_width, img_height)))
    return ret


def _build_seq(s):
    vals = [eval(x) for x in s.split(' ') if x.isdigit()]
    ret = []
    for islice in range(0, len(vals)-1, 2):
        ret.append((vals[islice], vals[islice] + vals[islice+1]))
    return ret


def _get_subimage(full_image, slices):
    ret = []
    for sl in slices:
        chunk = full_image[sl[0]:sl[1]+1]
        ret.extend(chunk)
    return np.array(ret)
