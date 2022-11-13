1. 개요
 - 머신 러닝을 이용하여 자신만의 프로젝트 만들기
 - 주제 : 머신 러닝을 이용한 Character LCD 상의 문자영역 추출

2. 주제
 - Text LCD 상의 Text영역 검출 및 Text 반전 영역 검출
<img src="https://user-images.githubusercontent.com/77335485/201528522-de9fa426-185b-4abc-9442-052e654e6256.png" width="800"> 

3. 개발환경
  - 개발 툴 : PyCharm 2021.1.2.
  - 개발 언어 : Python 3.8
  - 참고 소스 : https://github.com/ultralytics/yolov5
  - CUDA Version 11.2
  - cudnn : cudnn-11.2-windows-x64-v8.1.1.33
  - 라벨링 툴 : LabelImg
  -Yolo Version : YOLOv5

4. Data 학습 : 유튜브에서 취득
  - Tranin 영상 - 39
  - Validation 영상 - 7개
  
5. Data 학습 : 이미지에서 검출되어야 하는 영역에 대한 라벨링 실시(LabelImg 사용)
<img src="https://user-images.githubusercontent.com/77335485/201528677-7464cfe9-75c5-49b3-ae7b-24e1e34df466.png" width="800"> 

6. 결과 : epoch :50
<img src="https://user-images.githubusercontent.com/77335485/201528776-9b163872-bf79-479d-8179-3ee5eb0b7663.png" width="800"> 
