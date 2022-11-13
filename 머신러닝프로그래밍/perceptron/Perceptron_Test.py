
ort numpy as np

def OR_gate(x1, x2):
    w1 = 1
    w2 = 1
    b=-0.5
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1

def AND_gate(x1,x2):
    w1 = 1
    w2 = 1
    b = -1.5
    result = x1 * w1 + x2 * w2 + b
    if result < 0:
        return 0
    else:
        return 1

def NAND_gate(x1,x2):
    w1 = -1
    w2 = -1
    b = 1.5
    result = x1 * w1 + x2 * w2 + b

    if result < 0:
        return 0
    else :
        return 1
def XOR_gate(x1,x2):
     s1 = NAND_gate(x1, x2)
     s2 = OR_gate(x1, x2)
     y = AND_gate(s1, s2)
     return y

print(OR_gate(0, 0), OR_gate(0, 1), OR_gate(1, 0), OR_gate(1, 1))
print(AND_gate(0, 0), AND_gate(0, 1), AND_gate(1, 0), AND_gate(1, 1))
print(NAND_gate(0, 0), NAND_gate(0, 1), NAND_gate(1, 0), NAND_gate(1, 1))
print(XOR_gate(0, 0), XOR_gate(0, 1), XOR_gate(1, 0), XOR_gate(1, 1))

# OR ----------->
#                 AND ------------> XOR
# NAND---------->

# x1     x2     NAND     OR     y
#  0      0     1        0      0
#  1      0     1        1      1
#  0      1     1        1      1
#  1      1     0        1      0

#Multi-Layers Perceptron

from matplotlib import pyplot as plt
"""
#학습 데이터 생성 및 파라미터 초기화

# Neural Networks 2-2-1

# train data (XOR Problem)
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

# Intialization

# input - hidden layer
w1 = np.random.randn(2,2)
b1 = np.random.randn(1,2)

# hidden - output layer
w2 = np.random.randn(1,2)
b2 = np.random.randn(1)

# epoch
ep = 20000

# learning rate
lr = 1
mse = []

#신경망 순전파 출력 단계
# Neural Networks 2-2-1
for i in range(ep):

    E = np.array([])
    result = np.array([])

    for j in range(len(x)):
        Ha = np.array([])

        # feedforward
        # input - hidden layer
        for k in range(len(w1)):
            Ha = np.append(Ha, 1 / (1 + np.exp(-(np.sum(x[j] * w1[k]) + b1[0][k]))))

        # hideen - output layer
        Hb = 1 / (1 + np.exp(-(np.sum(Ha * w2) + b2)))

        # error
        E = np.append(E, y[j] - Hb)
        result = np.append(result, Hb)
# 신경망 역전파 업데이트
        # back-propagation
        # output - hidden layer
        alpha_2 = E[j] * Hb * (1 - Hb)

        # hidden - input layer
        alpha_1 = alpha_2 * Ha * (1 - Ha) * w2

        # update
        w2 = w2 + (lr * alpha_2 * Ha)
        b2 = b2 + lr * alpha_2

        w1 = w1 + np.ones((2, 2)) * lr * alpha_1 * x[j]
        b1 = b1 + lr * alpha_1

    print('EPOCH : %05d MSE : %04f RESULTS : 0 0 => %04f 0 1 => %04f 1 0 => %04f 1 1 => %04f'
          % (i, np.mean(E ** 2), result[0], result[1], result[2], result[3]))

    mse.append(np.mean(E ** 2))
#가시화
    # plot graph

    if i%100 == 0:
        plt.xlabel('EPOCH')
        plt.ylabel('MSE')
        plt.title('MLP TEST')
        plt.plot(mse)
        plt.show()
"""
###############################

import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# xor 문제 풀기 위한 입출력 정의
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device) # 입력 값
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)# 출력값

#다층 퍼셉트론 설계
#입력층 , 은닉층 1, 은닉층 2, 은닉층 3, 출력층으로 구성
model = nn.Sequential(
          nn.Linear(2, 10, bias=True), # input_layer = 2, hidden_layer1 = 10 은닉층 1
          nn.Sigmoid(),
          nn.Linear(10, 10, bias=True), # hidden_layer1 = 10, hidden_layer2 = 10 은닉층 2
          nn.Sigmoid(),
          nn.Linear(10, 10, bias=True), # hidden_layer2 = 10, hidden_layer3 = 10 은닉층 3
          nn.Sigmoid(),
          nn.Linear(10, 1, bias=True), # hidden_layer3 = 10, output_layer = 1 출력층
          nn.Sigmoid()
          ).to(device)

# 비용 함수와 옵타마이저 선언
# 비용 함수 : 원래의 값과 가장 오차가 작은 가설함수를 도출하기 위해 사용하는 함수
criterion = torch.nn.BCELoss().to(device) #비용함수
optimizer = torch.optim.SGD(model.parameters(), lr=1)  # modified learning rate from 0.1 to 1    옵티마이저
                                                        # #SDG: Stochastic Gradient Descent


#한 번의 epoch는 인공 신경망에서 전체 데이터 셋에 대해 forward pass/backward pass 과정을 거친 것을 말함.
# 즉, 전체 데이터 셋에 대해 한 번 학습을 완료한 상태
# 10001번의 에포크 수행. 각 에포크마다 역전파 수행됨
for epoch in range(10001):
    optimizer.zero_grad()
    # forward 연산
    hypothesis = model(X)

    # 비용 함수
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    # 100의 배수에 해당되는 에포크마다 비용을 출력
    if epoch % 100 == 0:
        print(epoch ,cost.item())



#예측값 확인
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())


