#2. 공간 도메인 필터링
#각 픽셀에 임의의 값을 더해 노이즈를 생성하고, 사용자로부터 Bilateral filtering을 위한 diameter, SigmaColor, SigmaSpace를 입력받아 노이즈를 제거하고
# 노이즈 제거 전후의 영상을 출력하시오. (다양한 파라미터 변화를 통해 영상이 어떻게 변화하는지 보고서에 넣으시오.)


import cv2
import numpy as np
#import matplotlib.pyplot as plt

cv2.namedWindow('window')

image = cv2.imread('../TestImage/lena.png').astype(np.float32)/255

noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0,1)

diameter = 0 #int
SigmaColor =0.0 # double
SigmaSpace = 0.0#double

bilat = cv2.bilateralFilter(noised,diameter,SigmaColor,SigmaSpace)

fiil_val = np.array([128,255,255],np.uint8)
bchange = False

def on_change(idx,value):
    fiil_val[idx] = value
    global  bchange
    bchange = True

fiil_val[0] = 0
fiil_val[1] = 0
fiil_val[2] = 0

cv2.imshow('original',noised)
cv2.createTrackbar('diameter','window',1,50,lambda v:
on_change(0,v))
cv2.createTrackbar('SigmaColor','window',0,100,lambda v:
on_change(1,v))
cv2.createTrackbar('SigmaSpace','window',0,20,lambda v:
on_change(2,v))

while True:
    diameter = fiil_val[0]- 25
    SigmaColor = fiil_val[1]/100.0
    SigmaSpace  = fiil_val[2]
    if bchange == True:
        print(diameter,SigmaColor,SigmaSpace)
        bilat = cv2.bilateralFilter(noised, diameter, SigmaColor, SigmaSpace)
        bchange = False

    text = 'diameter : ' + str(diameter)
    org = (5, 485)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bilat, text, org, font, 0.5, (255, 0, 0), 1)

    text = 'SigmaColor : ' + str(SigmaColor)
    org = (5, 505)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bilat, text, org, font, 0.5, (255, 0, 0), 1)

    text = 'SigmaSpace : ' + str(SigmaSpace)
    org = (5, 525)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bilat, text, org, font, 0.5, (255, 0, 0), 1)

    cv2.imshow('window',bilat)
    key = cv2.waitKey(1)

    if key ==27:
        break;
cv2.destroyAllWindows()
