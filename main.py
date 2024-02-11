import cv2
import numpy as np
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1 
    color = color or [int(c) for c in np.random.randint(0, 255, 3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1) 
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]  
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

modelo = attempt_load('yolov5s.pt')

device = torch.device('cpu')

source = 0

while True:
    cap = cv2.VideoCapture(source)
    ret, img0 = cap.read()

    img = img0[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = np.expand_dims(img, axis=0)

    img = torch.from_numpy(img).to(device)
    img = img.float()  
    img /= 255.0

    pred = modelo(img, augment=False)[0]

    pred = non_max_suppression(pred, 0.4, 0.5)

    for det in pred[0]:
        xyxy = det[:4] 
        label = '%s %.2f' % ('objeto', det[4])  
        plot_one_box(xyxy, img0, label=label)

    cv2.imshow('YOLOv5', img0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
