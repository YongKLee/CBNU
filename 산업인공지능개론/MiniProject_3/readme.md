1. 개요 
  1) 목표
   - 실습한 ResNet 모델을 전이학습으로 하는 방법을 사용하여, 현업에서 분류가 필요한 영상들을 학습 데이터로 하여 모델을 학습시키시오. 영상의 부류의 개수는 최소 2개 이상으로 하고, 부류별 데이터는 150장을 수집하여, 120장은 학습에 30장은 검증에 사용하시오.  (부류의 개수가 3개인 경우, 3x150 = 450장 필요, 동일한 대상에 대해서 여러 방향에서 사진을 찍어도 됨) 
  2) 주제
   - ResNet 모델을 전이학습 할 것
   - 주제 :  Remote-IO,  Server,  노트북, 산업용 Fanless 컴퓨터 영상을 학습 및 분류(4개의 부류) 
   - 수집 영상 수
     - Remote-IO  :  train  -> 125장, validation -> 49장
     -  1U Server  :  train  -> 125장, validation -> 49장
     -  노트북   :  train  -> 120장, validation -> 38장
     -  산업용 Fanless 컴퓨터   :  train  -> 132장, validation -> 60장
     
 2. 학습 영상 예시
 
 <img src="https://user-images.githubusercontent.com/77335485/201525244-211465f4-3501-490b-8ada-677baaff028c.png"  width="640" height="300">
 

3. 학습 결과

<img src="https://user-images.githubusercontent.com/77335485/201525501-28354538-bcd0-4689-884a-a4f23f91f4e7.png"  width="640" height="300">
