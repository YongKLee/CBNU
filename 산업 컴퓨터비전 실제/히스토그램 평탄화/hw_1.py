#1. 히스토그램 평탄화
#사용자로부터 R, G, B 중의 하나의 채널을 입력받고 입력받은 채널에 대한 히스토그램을 그리고 평탄화를 한 후에 그 영상을 출력하시오. (선택받은 채널 이외의 채널 값은 변화하지 않음)

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../TestImage/lena.png')

image_to_show = np.copy(img) #display 및 영상 처리용 image copy
chans = cv2.split(img) # image 채널 분리
colors = ("b", "g", "r") # rgb 3개 채널 정의

finish = False; # while 문 조건 초기화
color_ch = '' # 입력 받은 채널 index
seletecd = False # 원하는 키 입력 여부 flag
cv2.imshow('Result', image_to_show) # image view

while not finish:
    key = cv2.waitKey(0)
    if key == ord('b'): # keyboard b input
        sel_ch = 0
        color_ch = "Blue"
        seletecd = True
    elif key == ord('g'): # keyboard g input
        sel_ch = 1
        color_ch = "Green"
        seletecd = True
    elif key == ord('r'): # keyboard r input
        sel_ch = 2
        color_ch = "Red"
        seletecd = True
    elif key == 27: # esc key input , while escape
        finish = True
        seletecd = False
    else:
        seletecd = False

    if seletecd == True :
        image_to_show = np.copy(img)
        chans = cv2.split(image_to_show)
        hist, bins = np.histogram(chans[sel_ch], 256, [0, 255]) #입력받은 채널 히스토그램 생성
        eq_hist = cv2.equalizeHist(chans[sel_ch]) # 채널 평탄화
        hist_eq, bins_eq = np.histogram(eq_hist, 256, [0, 255]) # 평탄화된 채널 히스토그램 생성

        plt.figure()
        plt.subplot(121)
        plt.fill(hist) # 히스토그램 입력
        plt.xlabel('pixel value')
        plt.subplot(122)
        plt.fill(hist_eq) # 평탄화된 히스토그램 입력
        xlabel_str = 'eq pixel value(' + color_ch+')'
        plt.xlabel(xlabel_str)
        plt.tight_layout(True)
        plt.show()
        chans[sel_ch] = eq_hist
        image_to_show[:, :, sel_ch] = eq_hist
        cv2.imshow('Result', image_to_show) #선택된 채널 평탄화된 이미지 출력

cv2.destroyAllWindows()

