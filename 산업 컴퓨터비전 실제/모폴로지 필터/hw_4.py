# 4. 모폴로지 필터
# 영상을 이진화한 후에 사용자로부터 Erosion, Dilation, Opening, Closing에 대한 선택과 횟수를 입력받아서 해당 결과를 출력하시오.

import cv2
import numpy as np

image = cv2.imread('../TestImage/lena.png', 0)
_, binary = cv2.threshold(image, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

selected = 0
iter_num = 0
isChange = True
cv2.namedWindow('window')


def on_change(idx, value):
    global selected
    global iter_num
    global isChange
    isChange = True
    if idx == 0:
        selected = value
    elif idx == 1:
        iter_num = value


cv2.createTrackbar('selction', 'window', 0, 3, lambda v:
on_change(0, v))
cv2.createTrackbar('iterations', 'window', 0, 20, lambda v:
on_change(1, v))

display_img = np.zeros_like(binary)
display_img[binary > 0] = 255
cv2.imshow('binary ', display_img)

while True:
    if isChange:
        isChange = False
        print(iter_num)
        _, binary = cv2.threshold(image, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, (3, 3), iterations=iter_num)
        dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, (3, 3), iterations=iter_num)

        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                  iterations=iter_num)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                  iterations=iter_num)

        #        grad = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT,
        #                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        text = ''
        if selected == 0:
            display_img = np.zeros_like(eroded)
            display_img[eroded > 0] = 255
            text = 'eroded'
        elif selected == 1:
            display_img = np.zeros_like(dilated)
            display_img[dilated > 0] = 255
            text = 'dilated'
        elif selected == 2:
            display_img = np.zeros_like(opened)
            display_img[opened > 0] = 255
            text = 'opened'
        elif selected == 3:
            display_img = np.zeros_like(closed)
            display_img[closed > 0] = 255
            text = 'closed'


        cv2.imshow('window', display_img)

    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
