"""
trich xuat dac trung
model arcface
"""

import os
from packages.FaceRecognition import facemod
from packages.FaceRecognition.arcface import ArcFace
import numpy as np
import csv

def get_path(dir_path):
    """

    :param dir_path: thư mục chứa các class (người)
    :return: danhh sách các path dẫn đến thư mục của mỗi class
    """
    paths = []
    if os.path.isdir(dir_path): # kiểm tra dir_path có tồn tại
        names = os.listdir(dir_path)    # lấy danh sách các class trong thư mục
        paths = [os.path.join(dir_path, name) for name in names]    # lấy danh sách các path của các class
    return paths

def main():
    data_dir = "Dataset/FaceData/procced"
    names = []
    image_paths = get_path(data_dir)
    for path in image_paths:
        name = os.path.basename(path)
        names.append(name)
    print(names)
    dataset = facemod.get_dataset(data_dir)
    face_rec = ArcFace.ArcFace()
    # kiem tra co it nhat 1 anh trong moi class
    for cls in dataset:
        assert (len(cls.image_paths) > 0, 'Phai co it nhat 1 anh')

    paths, labels = facemod.get_image_paths_and_labels(dataset)
    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))
    # print(labels)
    namess = []
    for i in range(len(labels)):
        namess.append(names[labels[i]])
    # print(namess)
    # Tinh embeddings
    print("Calculating feature for image")
    nrof_images = len(paths)
    # emb_array = np.zeros((nrof_images, 512))
    for i in range(len(paths)):
        # print(paths[i])
        # print(labels[i])
        emb = face_rec.calc_emb(paths[i])
        print(emb.shape)
        with open("arc_emb.csv", 'w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['label', 'emb'])
            for j in range(len(labels)):
                writer.writerow([namess[j], emb])
main()