"""
Đăng kí khuôn mặt người dùng
+ Chụp ảnh mặt người dùng
+ Lưu vào CSDL
"""


import cv2
import sqlite3
import os
from packages.FaceDetection.centerface import CenterFace

def insertOrUpdate(id, name):
    """
    Cập nhật người dùng vào CSDL
    :param id:
    :param name:
    :return:
    """
    conn = sqlite3.connect("FaceBaseNew.db")
    cursor = conn.execute('SELECT * FROM People WHERE ID=' + str(id))
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
        break
    if isRecordExist == 1:
        cmd = "UPDATE people SET Name=' " + str(name) + " ' WHERE ID=" + str(id)
    else:
        cmd = "INSERT INTO people(ID,Name) Values(" + str(id) + ",' " + str(name) + " ' )"
    conn.execute(cmd)
    conn.commit()
    conn.close()

def main():
    # Khởi tạo
    FACE_SIZE = 160
    sample_num = 0  # bien dem so mau duoc chup
    class_id = input('ID: ')    # ID nguoi dung
    class_name = input('Name: ')    # ten nguoi dung
    output_dir = 'Dataset/FaceData/procced' # thu muc luu du lieu nguoi dung
    class_filename = str(class_name + class_id) # Tên của class người dùng == Tên thư mục người dùng
    output_class_dir = os.path.join(output_dir, class_filename) # Thư mục người dùng
    if not os.path.exists(output_class_dir):    # neu khong ton tai thu muc thi tao thu muc nguoi dung moi
        os.makedirs(output_class_dir)
    faceDetector = CenterFace()

    # bat dau qua trinh dang ky
    print("Start registing!")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        # Kẻ khung giữa màn hình để người dùng đưa mặt vào khu vực này
        centerH = frame.shape[0] // 2
        centerW = frame.shape[1] // 2
        sizeboxW = 300
        sizeboxH = 400
        cv2.rectangle(frame, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                      (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 1)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Có nên convert về ảnh xám?? model có lấy feature màu sắc???
        # Phát hiện khuôn mặt bằng face_detector
        dets, lms = faceDetector(frame, h, w, threshold=0.5)
        for det in dets:
            boxes = det[:4]
            cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (0, 0, 255), 2)
            face_scaled = cv2.resize(src=frame[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])], dsize=(FACE_SIZE, FACE_SIZE))
            cv2.imwrite(output_class_dir + '/' + class_filename + str(sample_num) + ".jpg", face_scaled)
            sample_num += 1
            print(sample_num)

        cv2.imshow('frame', frame)
        # Check xem có bấm q hoặc trên 100 ảnh sample thì thoát
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sample_num > 100:
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
