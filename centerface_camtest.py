"""

"""
import cv2
from packages.FaceDetection.centerface import CenterFace


def camera():
    cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()

    faceDetector = CenterFace()
    while True:
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        dets, lms = faceDetector(frame, h, w, threshold=0.5)
        # print(dets)
        for det in dets:
            boxes = det[:4]
            cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (0, 0, 255), 2)
        cv2.imshow('out', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


if __name__ == '__main__':
    camera()
