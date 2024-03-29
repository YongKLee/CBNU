import numpy as np
import cv2
import winsound
#물체 부류 이름 색깔 설정
classes = []
f=open('coco.names.txt','r')
classes=[line.strip() for line in f.readlines()]
colors=np.random.uniform(0,255,size=(len(classes),3))
##################
#Yolov3 모델을 읽고 출력층 알아내는 부분
yolo_model=cv2.dnn.readNet('yolov3.weights','yolov3.cfg') # 욜로 학습 데이터 읽어오기
#신경망 출력하는 3개층 알아내기 (yolo_82,yolo_94,yolo_106)
layer_names=yolo_model.getLayerNames()
out_layers=[layer_names[i[0]-1] for i in yolo_model.getUnconnectedOutLayers()]
####################
#웹캠을 통해 비디오 프레임을 처리하는 함수
def process_video(): # 비디오에서 침입자 검출해 알리기
    video=cv2.VideoCapture(0)# opencv로 0번 카메라로 영상 캡쳐 설정
    while video.isOpened():
        success,img=video.read()
        if success:
            height,width,channels=img.shape
            blob=cv2.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True,crop=False)#입력 영상을 블롭(blob) 객체로
                                                                                            # 만들어서 추론을 진행

            yolo_model.setInput(blob)
            output3=yolo_model.forward(out_layers)

            class_ids,confidences,boxes=[],[],[]
            for output in output3:
                for vec85 in output:
                    scores=vec85[5:]
                    class_id=np.argmax(scores)
                    confidence=scores[class_id]
                    if confidence>0.5: # 신뢰도가 50% 이상인 경우만 취함
                        centerx,centery=int(vec85[0]*width),int(vec85[1]*height) # [0,1] 표현을 영상 크기로 변환
                        w,h=int(vec85[2]*width),int(vec85[3]*height)
                        x,y=int(centerx-w/2),int(centery-h/2)
                        boxes.append([x,y,w,h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                    
            indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
                    
            for i in range(len(boxes)):
                if i in indexes:
                    x,y,w,h=boxes[i]
                    text=str(classes[class_ids[i]])+'%.3f'%confidences[i]
                    cv2.rectangle(img,(x,y),(x+w,y+h),colors[class_ids[i]],2)
                    cv2.putText(img,text,(x,y+30),cv2.FONT_HERSHEY_PLAIN,2,colors[class_ids[i]],2)

            cv2.imshow('Object detection',img)

            if 0 in class_ids: # 사람이 검출됨(0='person') class id 가 사람인 경우
                print('사람이 나타났다!!!')
                winsound.Beep(frequency=2000,duration=500) # Beep음 출력

        key=cv2.waitKey(1) & 0xFF
        if key==27: break

    video.release()
    cv2.destroyAllWindows()

process_video()
