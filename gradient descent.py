import numpy as np
import matplotlib.pyplot as plt

###########################################
def left_graph(X, Y, a_list, MSE_list, update_a, a_grad) : # 왼쪽 차트

    # A = np.linspace(-2, 6.5)
    A = np.linspace(min(min(a_list)-1, -2), max(max(a_list), 2.3 + abs(2.3 - min(min(a_list)-1, -2)), 6.5))

    mse_list = []
    for a_ in A:
        Y_pred = a_ * X + b
        error = Y_pred - Y
        mse = np.mean(error ** 2)
        mse_list.append(mse)

    plt.plot(A, mse_list)
    plt.scatter(a_list[:-1], MSE_list[:-1]) # 처음부터 마지막-1 번째 점
    plt.scatter(a_list[-1], MSE_list[-1], c='r') # 마지막 점

    # 접선 그리기 (feat by dajung)
    xx = np.array([a_list[-1]-max(0.09*(abs(a_list[-1])), 1.5), a_list[-1] + max(0.09*(abs(a_list[-1])), 1.5)])
    yy = a_grad * (xx - a_list[-1]) + MSE_list[-1] # 기울기가 a_grad이고, (a, MSE)를 지나는 직선의 방정식
    plt.plot(xx, yy)


    # 발산할 경우 경고 문구
    if len(MSE_list)>=2 and (MSE_list[-2] < MSE_list[-1]) : msg = "danger"
    else                                                  : msg = ""

    # 이전 점과 현재 점 화살표로
    plt.annotate(msg, ha='center', va='bottom', xytext=(a_list[-1], MSE_list[-1]), xy=(a_list[-1], MSE_list[-1]), arrowprops={'edgecolor': 'b', 'alpha': 0.5, 'arrowstyle': '->'}) # 화살표???
    plt.plot(a_list[-2:], MSE_list[-2:])

    # 현재 Epoch
    if len(a_list)>=2 : epoch_msg = f"Epoch : {len(a_list)-1}"
    else              : epoch_msg = "User's Initial value a"

    plt.xlabel(f"a ({epoch_msg})")
    plt.ylabel("MSE(cost)")
    plt.title(f"a * x + b = {a_list[-1]:,.2f} * x + {b:,.2f}\na : {a_list[-1]:,.2f},    a_grad : {a_grad:,.2f},    MSE : {MSE_list[-1]:,.2f}\n\nupdate_a(next a) :\n    a = a - lr * a_grad = {a_list[-1]:,.2f} - {lr:,.2f} * {a_grad:,.2f} = {update_a:,.2f}\n\n", fontsize=7, loc='left')

def right_graph(X, Y, Y_pred, a_list, MSE):  # 오른쪽 차트
    plt.scatter(X, Y)
    plt.scatter(X, Y_pred)
    plt.plot(X, Y_pred, c='r')
    plt.xlabel("study_time")
    plt.ylabel("score")

    Y_pred_round = np.round(Y_pred, 2)
    Y_round = np.round(Y, 2)
    err = np.round(Y_pred - Y, 2)
    err_squ = np.round((Y_pred - Y)**2, 2)
    plt.title(f"Y_prediction = {a_list[-1]:,.2f} * x + {b:,.2f} = {a_list[-1]:,.2f} * {X} + {b:,.2f}\nerror = Y_prediction - Y_real = {Y_pred_round} - {Y_round}\n         = {err}\nerror^2 = {err_squ}\nMSE = sum(error^2) / n = sum({err_squ}) / {n} = {MSE:,.2f}\n\n", fontsize=8, loc='left')
###########################################

X = np.array([2, 4, 6, 8])
Y = np.array([81, 93, 91, 97])

a = -2          # a : 초기 ax+b의 기울기 값(변경해 볼 것...)
b = 79          # 이건 변경하지 말 것...
lr = 0.02       # lr : learning rate(학습률, 혹은 stepsize라고 부르기도 함.)로 a, b를 얼만큼씩 업데이이트 시켜야 하는지 결정(너무 작으면 학습 속도가 느림, 너무 크면 발산할 수 있음... 따라서 적당한 값
epoch = 15      # 전체 데이터를 이용해 1번 학습한 것을 1 epoch함.
n = len(X)

a_list = []
a_grad_list = []
MSE_list = []

plt.figure(figsize=(11, 4.5 * (epoch+1))) # 그래프 종이의 크기(가로8, 세로30)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=1.2) # 그래프 간격

for i in range(1, epoch+2) :
    Y_pred = a * X + b

    error = Y_pred - Y
    MSE = np.mean(error**2)

    a_grad = (2/n) * np.sum(error * X)   # a_grad = 2 * np.mean(error * x)
    b_grad = (2/n) * np.sum(error)       # b_grad = 2 * np.mean(error)

    before_a = a
    a = a - lr * a_grad
    # b = b - lr * b_grad                # 이건 그냥 그대로 둘 것...


    # 시각화 데이터
    a_list.append(before_a)
    MSE_list.append(MSE)
    a_grad_list.append(a_grad)

    plt.subplot(epoch+1, 2, 2*i-1)
    left_graph(X, Y, a_list, MSE_list, a, a_grad)

    plt.subplot(epoch+1, 2, 2*i)
    right_graph(X, Y, Y_pred, a_list, MSE)


plt.show()
