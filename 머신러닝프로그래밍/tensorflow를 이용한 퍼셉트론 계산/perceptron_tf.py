import tensorflow as tf

# OR 데이터 구축
x = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]] # OR 입력값 (0,0), (0,1), (1,0), (1,1),
y = [[-1], [1], [1], [1]]  # x에 대응되는 OR 결과값 FALSE , TRUE, TRUE => -1, 1, 1, 1


# 전방 계산(식 (4.3))
def forward():
    s=tf.add(tf.matmul(x,w),b)
    o=tf.tanh(s)
    return o

# 손실 함수 정의
def loss():
    o=forward()
    return tf.reduce_mean((y-o)**2)


# 가중치 초기화
w=tf.Variable(tf.random.uniform([2, 1], -0.5, 0.5)) # weight 변수 초기화
                                                    # shape=(2,1),
                                                    # dtype= float32, numpy array -0.5 ~ 0.5 사이값의 난수로 초기화
print("w 초기값:",w)
b=tf.Variable(tf.zeros([1])) # bias 변수 초기화
                             # shape=(1,)
                             # 0으로 초기화
print("b 초기값:",b)

# 옵티마이저
opt=tf.keras.optimizers.SGD(learning_rate=0.1) # 스토케스트 경사 하강법(SGD Stochastic Gradient Descent)으로
                                               # 옵티마이저 생성해 opt 객체에 저장
                                               # 학습률에 해당하는 learning_rate는 0.1로 설정


# 500세대까지 학습(100세대마다 학습 정보 출력)
for i in range(500):
    opt.minimize(loss, var_list=[w,b])
    if(i%100==0): print('loss at epoch',i,'=',loss().numpy())

# 학습된 퍼셉트론으로 OR 데이터를 예측
o=forward()
print(o)
