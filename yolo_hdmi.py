import cv2
import numpy as np

# 1. Read model ONNX bang OpenCV DNN (sieu nhe cho Pi)
net = cv2.dnn.readNetFromONNX("yolov8n-pose.onnx")

# Cac cap diem ket noi khung xuong nguoi (COCO 17 keypoints)
PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # Mat, mui, tai
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Vai, tay, khuyp tay
    (5, 11), (6, 12), (11, 12),              # Than nguoi, hong
    (11, 13), (13, 15), (12, 14), (14, 16)   # Chân, dau goi, co chan
]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("--- Dang chay YOLO Pose ONNX tren Pi ---")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # 2. Tien xu ly anh
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward() # Tra ve ma tran chua bounding box + 17 keypoints

    # 3. Giai ma va ve khung xuong
    output = np.squeeze(outputs[0]).T # Đổi chiều về [N, 56]

    for detection in output:
        score = detection[4] # Do tin cay nhan dien nguoi
        if score > 0.5:
            keypoints = detection[5:] # 17 diem keypoints (x, y, confidence)
            pts = []

            # Lay toan bo 17 diem khop
            for i in range(17):
                kx = int(keypoints[i*3] * w / 640)
                ky = int(keypoints[i*3 + 1] * h / 640)
                kconf = keypoints[i*3 + 2]
                pts.append((kx, ky, kconf))

                # Ve diem khop mau xanh la
                if kconf > 0.5:
                    cv2.circle(frame, (kx, ky), 3, (0, 255, 0), -1)

            # Noi cac duong xuong mau do
            for p1, p2 in PAIRS:
                if pts[p1][2] > 0.5 and pts[p2][2] > 0.5:
                    cv2.line(frame, (pts[p1][0], pts[p1][1]), (pts[p2][0], pts[p2][1]), (0, 0, 255), 2)

    # 4. Hien thi
    cv2.imshow("YOLO Pose HDMI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()