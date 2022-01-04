import cv2
import numpy as np

min_w = 90
min_h = 90
# 检测线的高度
line_height = 550

def center(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture("C:\\Users\\lms13\\Desktop\\cv_imgs\\video.mp4")
bgsubmog = cv2.createBackgroundSubtractorMOG2()
# offset
offset = 7
# 统计车的数量
carno = 0
# 形态学kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# 存放有效车辆的数组
cars = []
while True:
    ret, frame = cap.read()
    if ret:
        # 灰度
grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 去噪（高斯）
blur = cv2.GaussianBlur(grey, (3, 3), 5)
        # 去背影
mask = bgsubmog.apply(blur)
        # 腐蚀 去掉图中小块
erode = cv2.erode(mask, kernel)
        # 膨胀
dilate = cv2.dilate(erode, kernel, iterations=3)
        # 闭操作 去掉物体内部的小块
close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
        close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 画一条检测线
cv2.line(frame, (10, line_height), (1200, line_height), (255, 255, 0), 3)
        for index, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            # 对车辆宽高进行判断 是否是有效车辆
isValid = (w >= min_w) and (h >= min_h)
            if not isValid:
                continue
# 到这里都为有效的车
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cpoint = center(x, y, w, h)
            cars.append(cpoint)
            for x, y in cars:
                if (line_height - offset) < y < (line_height + offset):
                    carno += 1
cars.remove((x, y))
        cv2.putText(frame, "Cars Counts: " + str(carno), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
        cv2.imshow("erode", dilate)
        cv2.imshow("video", frame)
        key = cv2.waitKey(40)
        if key == 27:  # ESC退出
break
cap.release()
cv2.destroyAllWindows()
