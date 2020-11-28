"""
dung faiss cho face recognition
"""

from packages.FaceRecognition.arcface import ArcFace
from packages.FaceRecognition import facemod
import os
import numpy as np
import pandas as pd
import faiss

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
df.to_csv('out.csv', index=False)

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

# searching
new_emb = face_rec.calc_emb('face4.png')
new_emb = np.array(new_emb).astype('float32').reshape(1,512)
# print(new_emb)
k = 5
D, I = index.search(new_emb, k)
print(I)
x = 0
for i in range(4):
    print(I[0][i])
    x += I[0][i]
x = int(x/5)
print(x)
print(namess[x])
print(D)