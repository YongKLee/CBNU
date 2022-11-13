1)	DFT를 통해서 영상을 주파수 도메인으로 바꿔서 출력 한 후에 사용자로부터 반지름을 입력받아서 그 크기만큼의 원을 그린 후에 DFT 결과에 곱해준 후에 IDFT를 해서 필터링된 영상을 출력하시오. 사용자로부터 Low pass인지 High Pass인지를 입력받아 Low pass면 원 안을 통과시키고, High Pass면 원 바깥을 통과시키도록 하시오.

2)	테스트 영상 : testimage\\lena.png

3)	실행 결과
-	Trackbar로 입력 옵션을 구현하였다. Low pass filter는 0으로 옵션을 설정했고 High pass filter는 1로 옵션을 구현하였다.

(1)	주파수 영상

<img src="https://user-images.githubusercontent.com/77335485/201529901-dfc86f05-dae4-46f2-abe6-bd6769384c2e.png"  width="640" height="300">

(2)	Low pass filter

<img src="https://user-images.githubusercontent.com/77335485/201529983-97ec9456-ef4f-48c3-a307-74f710f7f7bb.png"  width="640" height="300">

(3)	High pass filter

<img src="https://user-images.githubusercontent.com/77335485/201530011-e51937f1-47ce-4154-b382-5b4ecb1bb390.png"  width="640" height="300">
