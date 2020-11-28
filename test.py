from packages.FaceRecognition.arcface import ArcFace
from packages.FaceRecognition import facemod
import os
import numpy as np
import pandas as pd
import faiss
from imutils.video import VideoStream
import imutils
import cv2
from packages.FaceDetection.centerface import CenterFace

def putText(frame, name, best_class_probabilities, text_x, text_y):
    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (255, 255, 0), thickness=1, lineType=2)
    cv2.putText(frame, str(round(best_class_probabilities[0], 3)),
                           (text_x, text_y + 15),
                           cv2.FONT_HERSHEY_COMPLEX_SMALL,
                           1, (255, 0, 255), thickness=1, lineType=2)

face_rec = ArcFace.ArcFace()
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


data_dir = "Dataset/FaceData/p"
names = []
image_paths = get_path(data_dir)
for path in image_paths:
    name = os.path.basename(path)
    names.append(name)
print(names)
dataset = facemod.get_dataset(data_dir)
# kiem tra co it nhat 1 anh trong moi class
for cls in dataset:
    assert (len(cls.image_paths) > 0, 'Phai co it nhat 1 anh')
paths, labels = facemod.get_image_paths_and_labels(dataset)
print('Number of classes: %d' % len(dataset))
print('Number of images: %d' % len(paths))

namess = []
for i in range(len(labels)):
    namess.append(names[labels[i]])
print(namess)


emb = face_rec._calc_emb_list(paths)
emb = np.array(emb)
emb = emb.tolist()

# print(emb)

thisdict = {
    'name': namess,
    'emb': emb
}
df = pd.DataFrame(data=thisdict)
# print(df)

values = df.values
name = values[:, 0]
emb = []
for i in range(len(values[:, 1])):
    emb.append(np.array(values[i, 1]))
emb = np.array(emb).astype('float32')

# Building Index
index = faiss.IndexFlatL2(512)
print(index.is_trained)
index.add(emb)
print(index.ntotal)


cap = VideoStream(src=0).start()
faceDetector = CenterFace()
INPUT_IMAGE_SIZE = 112
while True:
    frame = cap.read()
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    bounding_boxes, _ = faceDetector(frame, h, w, threshold=0.5)
    faces_found = bounding_boxes.shape[0]
    try:
        if faces_found > 0:
            print('facefound: ', faces_found)
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]

                # print(bb[i][3] - bb[i][1])
                # print(frame.shape[0])
                # print((bb[i][3] - bb[i][1]) / frame.shape[0])
                if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0:
                    scaled = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)

                    # searching
                    new_emb = face_rec.calc_emb(scaled)
                    new_emb = np.array(new_emb).astype('float32').reshape(1,512)
                    # print(new_emb)

                    D, I = index.search(new_emb, 5)
                    print(D)
                    x = 0
                    for i in range(4):
                        # print(I[0][i])
                        x += I[0][i]
                    x = int(x/5)
                    # print(x)
                    print(namess[x])
                    # print(D)


                    cv2.putText(frame, namess[x], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    except:
        pass
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()