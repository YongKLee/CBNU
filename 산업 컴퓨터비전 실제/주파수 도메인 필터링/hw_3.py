# 3. 주파수 도메인 필터링
# DFT를 통해서 영상을 주파수 도메인으로 바꿔서 출력 한 후에 사용자로부터 반지름을 입력받아서 그 크기만큼의 원을 그린 후에 DFT 결과에 곱해준 후에 IDFT를 해서 필터링된 영상을 출력하시오.
# 사용자로부터 Low pass인지 High Pass인지를 입력받아 Low pass면 원 안을 통과시키고, High Pass면 원 바깥을 통과시키도록 하시오.


import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../TestImage/lena.png', 0).astype(np.float32) / 255


radius = 0

cv2.namedWindow('window')
filter_opt = 0  # lowpass : 0 high pass : 1
bchange = True


def on_change(idx, value):
    global radius
    global filter_opt
    if idx == 0:
        radius = value
    elif idx == 1:
        filter_opt = value
    global bchange
    bchange = True

cv2.createTrackbar('radius', 'window', 0, 100, lambda v:
on_change(0, v))
cv2.createTrackbar('Low/High', 'window', 0, 1, lambda v:
on_change(1, v))

while True:
    if bchange == True:
        bchange = False

        # DFT
        fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
        fft_shifted = np.fft.fftshift(fft, axes=[0, 1])
        fft_shifted_high = np.fft.fftshift(fft, axes=[0, 1])

        magnitude = cv2.magnitude(fft_shifted[:, :, 0], fft_shifted[:, :, 1])
        magnitude = np.log(magnitude)

        plt.figure()

        plt.imshow(magnitude,cmap='gray')
        plt.show()

        mask_low = np.zeros(fft.shape, np.uint8)  # low pass
        cy = mask_low.shape[0] // 2
        cx = mask_low.shape[1] // 2
        cv2.circle(mask_low, (cx, cy), radius, (1, 1, 1), -1)[0]

        mask_high = np.ones(fft.shape, np.uint8)  # high pass
        cy = mask_high.shape[0] // 2
        cx = mask_high.shape[1] // 2
        cv2.circle(mask_high, (cx, cy), radius, (0, 0, 0), -1)[0]

        fft_shifted *= mask_low
        fft_shifted_high *= mask_high

        fft = np.fft.ifftshift(fft_shifted, axes=[0, 1])
        fft_high = np.fft.ifftshift(fft_shifted_high, axes=[0, 1])

        filtered_low = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        filtered_high = cv2.idft(fft_high, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        if filter_opt == 1:
            cv2.imshow('window', filtered_high)
        else:
            cv2.imshow('window', filtered_low)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
