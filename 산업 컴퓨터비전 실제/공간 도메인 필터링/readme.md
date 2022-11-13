1. 주제 : 각 픽셀에 임의의 값을 더해 노이즈를 생성하고, 사용자로부터 Bilateral filtering을 위한 diameter, SigmaColor, SigmaSpace를 입력받아 노이즈를 제거하고 노이즈 제거 전후의 영상을 출력테

2. 테스트 영상 : testimage\\lena.png
3. 
4. 실행 결과

-	Bilateral filtering의 파라메터인 diameter, SigmaColor, SigmaSpace는 각각 Trackbar로 입력이 가능하게 구성하였다. 
  Trackbar로 소수점, 음수 영역을 구현하기 쉽지 않아 diameter Trackbar의 범위는 0-50으로 설정하였고 실제 diameter값에는 Trackbar값에 -25를 적용하였다. 
  SigmaColor Trackbar의 범위는 0-100으로 설정하였고 실제 SigmaColor 값에는 Trackbar값을 100으로 나누어 적용하였다. 
  SigmaSpace Trackbar의 범위는 0-20으로 설정하였고 실제 SigmaSpace 값에는 Trackbar값을 그대로 적용하였다.

-	Diameter 값이 양수로 증가할수록 영상의 노이즈가 없어지고 경계가 많이 무너지는 모습을 볼 수 있다. 

-	SigmaColor, SigmaSpace 값이 증가할수록 영상의 노이즈가 없어지고 경계가 많이 무너지는 모습을 볼 수 있다. 일정한 값이상이 넘으면 영상을 판독하기 힘들 정도로 영상이 뭉개짐을 볼 수 있었다.


(1)	Diameter : -1. SigmaSpace : 0.3, SigmaSpace:10
-	Diameter가 음수일 경우 SigmaSpace을 적용 받는다.

<img src="https://user-images.githubusercontent.com/77335485/201529570-127157df-9a24-4c03-a688-7f1faf341b81.png"  width="640" height="300">
 
(2)	Diameter : 3. SigmaSpace : 0.3, SigmaSpace:10

<img src="https://user-images.githubusercontent.com/77335485/201529652-f6eb11ea-4218-4d87-8a92-8b245d560196.png"  width="640" height="300">

(3)	Diameter : 22. SigmaSpace : 0.3, SigmaSpace:10

<img src="https://user-images.githubusercontent.com/77335485/201529725-4fa24ea9-db13-4292-86d1-394da6be5a4d.png"  width="640" height="300">



