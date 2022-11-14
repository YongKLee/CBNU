import numpy as np
import cv2
from scipy.ndimage import label
from matplotlib import pyplot as plt
from skimage import morphology
def hough_circle_detection(img, imgray , min_r, max_r):
    dst = img.copy()
    gray = imgray.copy()
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
#             cv2.HoughCircles(검출 이미지, 검출 방법, 해상도 비율, 최소 거리, 캐니 엣지 임곗값, 중심 임곗값, 최소 반지름, 최대 반지름)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=60, param2=50, minRadius=min_r, maxRadius=max_r)
    for i in circles[0,:]:
        cv2.circle(dst, (i[0], i[1]), i[2], (255, 255, 255), 5)
        print(i[0],i[1],i[2])
    return dst

def watershed(img,imgray):

    blur = cv2.GaussianBlur(imgray,(3,3),0)
    ret, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations =2)

    border = cv2.dilate(opening, kernel, iterations=3)

    erode_img = cv2.erode(border, None)
    border = border - erode_img

    # distance transform을 적용하면 중심으로 부터 외곽 이미지 추출(Skeleton Image)
    # 즉, 중심으로 부터 점점 옅어져 가는 영상.
    # 그 결과에 thresh를 이용하여 이진화
    dt = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    #거리값을 0~255로 범위 정규화
    dt = ((dt-dt.min())/(dt.max() - dt.min())*255).astype(np.uint8)
    #dt = (dt / (dt.max() - dt.min()) * 255).astype(np.uint8)

    #ret, dt = cv2.threshold(dt, 150, 255, cv2.THRESH_BINARY)
    print(dt.max())
    ret, dt_th = cv2.threshold(dt, 0.5 * dt.max(), 255,cv2.THRESH_BINARY)


    marker, ncc = label(dt_th)
    print(ncc)

    marker[border == 255] = 255
    marker = marker.astype(np.int32)

    cv2.watershed(img, marker)

    marker[marker==-1] = 0

    marker = marker.astype(np.uint8)
    marker = 255 - marker

    marker[marker != 255] = 0

    img[marker ==255] = (0,0,255)

    images = [imgray, blur,    thr,      opening,       erode_img,   border,     dt,    dt_th     , marker]
    titles = ['Gray', 'Blur', 'Binary', 'Morph(open)', 'erode_img', 'border',  'dt',    'dt(binary)',      'Marker']

    plt.figure()
    img_num = len(images)
    for i in range(img_num):
        plt.subplot(2, 5, i + 1), plt.imshow(images[i],cmap='gray'), plt.title(titles[i]), plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()

    return img

#img = cv2.imread('../coin_image/3058386_5301099_1 (1).jpg')
img = cv2.imread('../coin_image/quarters_dimes_pennies.png')
#img = cv2.imread('../coin_image/coin_sample.png')

img_original= img.copy()

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_mean_value=np.mean(imgray)
print(img_mean_value)
cv2.imshow('imgray', imgray)
if img_mean_value < 100:
    mask = imgray > img_mean_value
    coin_mask_clean = morphology.remove_small_objects(mask)
    coin_mask_clean = ~morphology.remove_small_objects(~coin_mask_clean)
    coin_mask_clean01 = morphology.remove_small_objects(mask)
    imgray = ~imgray

    imgray[coin_mask_clean01 == 0] = 255


else:
    imgray_mask = ~imgray
    mask = imgray_mask > (255 - img_mean_value)
    coin_mask_clean = morphology.remove_small_objects(mask)
    coin_mask_clean = ~morphology.remove_small_objects(~coin_mask_clean)
    coin_mask_clean01 = morphology.remove_small_objects(mask)

    imgray[coin_mask_clean01 == 0] = 255


cv2.imshow('imgray1', imgray)
hou_img = hough_circle_detection(img,imgray,20,150)
cv2.imshow('hough_circle_detection',hou_img)

watershed_img = watershed(img,imgray)
cv2.imshow('original',img_original)
cv2.imshow('watershed',watershed_img)


cv2.waitKey(0)

cv2.destroyAllWindows()