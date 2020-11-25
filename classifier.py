"""
Trích xuất đặc trưng mỗi class người dùng

"""

import tensorflow as tf
import numpy as np
from packages.FaceRecognition import facemod
import math
import os
import pickle
from sklearn.svm import SVC

def main():

    data_dir = "Dataset/FaceData/procced"
    model = "packages/FaceRecognition/Models/20180402-114759.pb"
    classifier_filename = "packages/FaceRecognition/Models/facemodel.pkl"
    batch_size = 1000
    image_size = 160
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            np.random.seed(seed=666)
            dataset = facemod.get_dataset(data_dir)

            # kiểm tra có ít nhất 1 ảnh trong mỗi class
            for cls in dataset:
                assert (len(cls.image_paths) > 0, 'Phải có ít nhất 1 ảnh')

            paths, labels = facemod.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load model
            print('Loading feature extraction model')
            facemod.load_model(model)

            # Input và Output Tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Tính embeddings
            print('Calculating feature for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facemod.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                # print(emb_array)
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            # train
            print('Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)
            # Tạo danh sách class
            class_names = [cls.name.replace('_', ' ') for cls in dataset]
            # Lưu model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Đã lưu model vào "%s"' % classifier_filename_exp)
main()