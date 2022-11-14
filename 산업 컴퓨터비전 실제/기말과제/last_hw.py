#14주차 강의까지 배운 내용들을 활용해서 특징 추출, 대응점 탐색, 옵티컬 플로우, 파노라마 스티칭, 카메라 캘리브레이션, 스테레오 매칭 등을 적용

import cv2

import numpy as np

def place_middle(number, new_size):
    h, w = number.shape[:2]
    big = max(h, w)
    square = np.full((big, big), 255, np.float32)  # 실수 자료형

    dx, dy = np.subtract(big, (w,h))//2
    square[dy:dy + h, dx:dx + w] = number
    return cv2.resize(square, new_size).flatten()  # 크기변경 및 벡터변환 후 반환

def find_value_position(img, direct):
    project = cv2.reduce(img, direct, cv2.REDUCE_AVG).ravel()
    p0, p1 = -1, -1                                                 # 초기값
    len = project.shape[0]                                   # 전체 길이
    for i in range(len):
        if p0 < 0 and project[i] < 250: p0 = i
        if p1 < 0 and project[len-i-1] < 250 : p1 = len-i-1
    return p0, p1

def find_number(part):
    x0, x1 = find_value_position(part, 0)  # 수직 투영
    y0, y1 = find_value_position(part, 1)  # 수평 투영
    return part[y0:y1, x0:x1]


def kNN_train(train_fname, nclass, nsample):
    size = (40, 40)  # 숫자 영상 크기
    train_img = cv2.imread(train_fname, cv2.IMREAD_GRAYSCALE)  # 학습 영상 적재
    h, w = train_img.shape[:2]
    dy = h % size[1]// 2
    dx = w % size[0]// 2
    train_img = train_img[dy:h-dy-1, dx:w-dx-1]             # 학습 영상 여백 제거
    cv2.threshold(train_img, 32, 255, cv2.THRESH_BINARY, train_img)

    cells = [np.hsplit(row, nsample) for row in np.vsplit(train_img, nclass)]
    nums = [find_number(c) for c in np.reshape(cells, (-1, 40,40))]
    trainData = np.array([place_middle(n, size) for n in nums])
    labels = np.array([i for i in range(nclass) for j in range(nsample)], np.float32)

    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, labels)  # k-NN 학습 수행
    return knn

def save_digit_image(src,filepath):
    cv2.imwrite(filepath, src)

def MouseLeftClick(event, x, y, flags, param):
	# 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
    if event == cv2.EVENT_LBUTTONDOWN:
        print('마우스 클릭 : ',x,y)


# number 학습
K1 = 10

nknn = kNN_train("../lcd_image/train_numbers.png", 10, 20)


img = cv2.imread('../lcd_image/main_vc1.jpg')
#img = cv2.imread('../lcd_image/213.jpg')



img_copy = img.copy()
img_gray= cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

#노이즈 제거
img_copy = cv2.GaussianBlur(img_gray, (7, 7), 0)

# LCD 영역 검출
return_value, threshold_image = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
im2, contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

contours_min = np.argmin(contours[0], axis = 0)
contours_max = np.argmax(contours[0], axis = 0)
x_min = contours[0][contours_min[0][0]][0][0]
y_min = contours[0][contours_min[0][1]][0][1]
x_max = contours[0][contours_max[0][0]][0][0]
y_max = contours[0][contours_max[0][1]][0][1]
print("x-Min =", x_min)
print("y-Min =", y_min)
print("x-Max =", x_max)
print("y-Max =", y_max)


# LCD 역영 추출
margin = 10
lcd_img = img_gray[y_min + margin: y_max - margin, x_min + margin: x_max - margin]


return_value, th_lcd_image = cv2.threshold(lcd_img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 숫자가 있는 영역
x_digit_min = 760
x_digit_max = 935
y_digit_min = 190
y_digit_max = 275

# 숫자 영역 추출
digit_img = th_lcd_image[y_digit_min:y_digit_max, x_digit_min: x_digit_max]

#숫자 분리
fcimg, contours, hierarchy = cv2.findContours(digit_img.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


digitCnts = []
roi_imgs =[]
contours.reverse()
i = 0;
for cnt in contours:
    area = cv2.contourArea(cnt)
    [x, y, w, h] = cv2.boundingRect(cnt)

    print(x, y, w, h)
    # if the contour is sufficiently large, it must be a digit
    if w >= 15 and (h >= 30 and h <= 80) and x > 0 and y > 0 and i <4:
        digitCnts.append(cnt)
        each_digit_img = digit_img[y:y+h, x: x+w]
        roi_imgs.append(each_digit_img)
        cv2.rectangle(digit_img, (x, y), (x + w, y + h), (0, 0, 255))
    i =i+1



#숫자 인식
numbers = [find_number(cell) for cell in roi_imgs]
datas = [place_middle(num,(40,40)) for num in numbers]
datas = np.reshape(datas,(len(datas),-1))

_, resp1, _, _ = nknn.findNearest(datas, K1)
resp1 = resp1.flatten().astype('int')

print(resp1)


cv2.rectangle(img,(x_min ,y_min ),(x_max ,y_max ),(0,0,255),2)
cv2.rectangle(img,(x_digit_min+x_min+margin,y_digit_min+y_min+margin),(x_digit_max+x_min+margin,y_digit_max+y_min+margin),(0,255,0),2)

font=cv2.FONT_HERSHEY_TRIPLEX
cv2.putText(img,str(resp1),(x_digit_min+x_min+margin,y_digit_max+y_min+50),font,2,(255,0,0),2)

cv2.imshow('result', img)
cv2.waitKey()
cv2.destroyAllWindows()

