import os
import sys
import cv2 as cv
import numpy as np


def cal_cosine_dist(img1, img2):
    h, w, c = img1.shape
    img1 = img1.reshape(h * w * c)
    img2 = img2.reshape(h * w * c)
    dist = np.dot(img1, img2) / (np.linalg.norm(img1) * np.linalg.norm(img2))
    return dist


def cal_eul_dist(pos1, pos2):
    pos1 = np.array(pos1, dtype=np.float)
    pos2 = np.array(pos2, dtype=np.float)
    return np.linalg.norm(pos1 - pos2)


def cal_dist(img1, img2, method='Euclidean'):
    """
    计算图像img1与img2的距离
    :param img1: numpy矩阵
    :param img2: numpy矩阵
    :param method: 距离度量方法
    :return:
    """
    h, w, c = img1.shape
    if img1.shape != img2.shape:
        print(f'img1.shape={img1.shape}, img2.shape={img2.shape}, not equal!')
        return sys.maxsize
    if method == 'Euclidean':
        dist = np.sqrt(np.sum(np.square(img1 - img2)) / (h * w * c))
    if method == 'cosine':
        dist = cal_cosine_dist(img1, img2)
    return dist


def find_best_match(img1, img2):
    """
    :param img1: large image
    :param img2: small image
    :return:
    """
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    dist_list = []
    min_dist = sys.maxsize
    min_pos = [0, 0]
    for i in range(h1 - h2):
        for j in range(w1 - w2):
            img1_crop = img1[i:i + h2, j:j + w2, :]
            dist = cal_dist(img1_crop, img2)
            if dist < min_dist:
                min_dist = dist
                min_pos = [i, j]
            print(f'({i},{j}), dist={dist}')
            dist_list.append(dist)
    return min_pos, min_dist


def get_content_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        line = f.readline()
    content_list = line.split(' ')
    return content_list


def test():
    txt_path = '../data/Label/2_sar_1.txt'
    print(get_content_from_txt(txt_path))


if __name__ == '__main__':
    test()
