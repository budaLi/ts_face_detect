from urllib import request
import os
import sys
import tarfile
import glob
import pickle
import numpy as np
import cv2


def download_and_uncompress_tarball(tarball_url, dataset_dir):
    """
    下载cifar10数据集
    """
    filename = tarball_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    filepath, _ = request.urlretrieve(tarball_url, filepath, _progress)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


classification = ['airplane',
                  'automobile',
                  'bird',
                  'cat',
                  'deer',
                  'dog',
                  'frog',
                  'horse',
                  'ship',
                  'truck']

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def create_date():
    """
    对数据集解压 并将其图片和labels存储
    :return:
    """
    folders = r'C:\Users\lenovo\PycharmProjects\ts_face_detect\data_manager\data\cifar-10-batches-py'
    trfiles = glob.glob(folders + r"\test*")
    data  = []
    labels = []
    for file in trfiles:
        dt = unpickle(file)
        data += list(dt[b"data"])
        labels += list(dt[b"labels"])

    print(labels)

    imgs = np.reshape(data, [-1, 3, 32, 32])

    for i in range(imgs.shape[0]):
        im_data = imgs[i, ...]
        #将数据格式进行转换
        im_data = np.transpose(im_data, [1, 2, 0])
        #将 rgb格式转换为 bgr
        im_data = cv2.cvtColor(im_data, cv2.COLOR_RGB2BGR)

        f = "{}\{}".format(r"C:\Users\lenovo\PycharmProjects\ts_face_detect\data_manager\data\image\test", classification[labels[i]])

        if not os.path.exists(f):
            #这里使用makedirs创建文件夹
            os.makedirs(f)

        cv2.imwrite("{}\{}.jpg".format(f, str(i)), im_data)


if __name__ == '__main__':
    # #数据下载地址
    # DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    # #数据存放目录
    # DATA_DIR = 'data'
    # download_and_uncompress_tarball(DATA_URL,DATA_DIR)

    create_date()












