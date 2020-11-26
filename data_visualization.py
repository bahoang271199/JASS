from packages.FaceRecognition.arcface import ArcFace
from packages.FaceRecognition import facemod
import os
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
face_rec = ArcFace.ArcFace()
import plotly.express as px

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
# print(names)
dataset = facemod.get_dataset(data_dir)
# kiem tra co it nhat 1 anh trong moi class
for cls in dataset:
    assert (len(cls.image_paths) > 0, 'Phai co it nhat 1 anh')
paths, labels = facemod.get_image_paths_and_labels(dataset)
# print('Number of classes: %d' % len(dataset))
# print('Number of images: %d' % len(paths))

namess = []
for i in range(len(labels)):
    namess.append(names[labels[i]])
# print(namess)


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

df = pd.DataFrame(data = emb )
df.insert(0, 'name', namess)
# print(df.columns)
columns_name = df.columns
# print(columns_name[11:511])
df = df.drop(columns=columns_name[11:490], axis=1)
print(df)
# # # print(emb[0][0])
# # fig = px.parallel_coordinates(df, color='name', color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)
# # fig.show()
parallel_coordinates(df, 'name')
plt.show()