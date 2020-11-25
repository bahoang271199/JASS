"""
Nhận diện bằng camera
"""

import cv2
import pickle
import tensorflow as tf
from packages.FaceRecognition import facemod
import collections
from imutils.video import VideoStream
import sys
from packages.FaceDetection.centerface import CenterFace
import imutils
import numpy as np


def putText(frame, name, best_class_probabilities, text_x, text_y):
    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (255, 255, 0), thickness=1, lineType=2)
    cv2.putText(frame, str(round(best_class_probabilities[0], 3)),
                           (text_x, text_y + 15),
                           cv2.FONT_HERSHEY_COMPLEX_SMALL,
                           1, (255, 0, 255), thickness=1, lineType=2)

def main():
    # Khởi tạo các tham số
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = "packages/FaceRecognition/Models/facemodel.pkl"
    FACE_MODEL_PATH = "packages/FaceRecognition/Models/20180402-114759.pb"

    # Load Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Load classifier success")

    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # load model
            print("Loading feature extraction model")
            facemod.load_model(FACE_MODEL_PATH)
            # Input và Output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            people_detected = set()
            person_detected = collections.Counter()

            cap = VideoStream(src=0).start()
            centerface = CenterFace()

            while True:
                frame = cap.read()
                frame = imutils.resize(frame, width=700)
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                bounding_boxes, _ = centerface(frame, h, w, threshold=0.5)
                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            print(bb[i][3] - bb[i][1])
                            print(frame.shape[0])
                            print((bb[i][3] - bb[i][1]) / frame.shape[0])
                            if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0:
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = facemod.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                                if best_class_probabilities > 0.8:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20

                                    name = class_names[best_class_indices[0]]
                                    putText(frame, name, best_class_probabilities, text_x, text_y)
                                    person_detected[best_name] += 1
                                else:
                                    name = "vailluondaucatmoiz"
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20

                                    putText(frame, name, best_class_probabilities, text_x, text_y)
                                    person_detected[best_name] += 1
                except:
                    pass

                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # cap.release()
            cv2.destroyAllWindows()

main()