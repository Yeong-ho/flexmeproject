import numpy as np
from datetime import datetime

loaded_data = np.loadtxt('./sps.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[ :, 1:]
t_data = loaded_data[ :, [0]]

np.random.seed(0)

W = np.random.rand(4,1)  # 4X1 행렬
b = np.random.rand(1)  

def loss_func(x, t):
    
    y = np.dot(x,W) + b # sigmoid 일떄는 z로 변경
    # sigmoid : y= 1/ (1+np.exp(-z))
    return ( np.sum( (t - y)**2 ) ) / ( len(x) )

def numerical_derivative(f, x):
    delta_x = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index        
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # f(x+delta_x)
        
        x[idx] = float(tmp_val) - delta_x 
        fx2 = f(x) # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)
        
        x[idx] = tmp_val 
        it.iternext()   
        
    return grad

# 손실함수 값 계산 함수
# 입력변수 x, t : numpy type
def error_val(x, t):
    y = np.dot(x,W) + b
    
    return ( np.sum( (t - y)**2 ) ) / ( len(x) )


learning_rate = 1e-3  # step당 곱하는 변화값

f = lambda x : loss_func(x_data,t_data)

print("Initial error value = ", error_val(x_data, t_data), "Initial W = ", W, "\n", ", b = ", b )

start_time = datetime.now()

for step in  range(50001):    # 3만번 반복수행
    
    W -= learning_rate * numerical_derivative(f, W)
    
    b -= learning_rate * numerical_derivative(f, b)
   
    if (step % 500 == 0):
        print("step = ", step, "error value = ", error_val(x_data, t_data) )
        
end_time = datetime.now()
        
print("")
print("Elapsed Time => ", end_time - start_time)

print("RESULTS W: ",W)
print("RESULTS B: ",b)

# 학습을 마친 후, 임의의 데이터에 대해 미래 값 예측 함수
# 입력변수 x : numpy type

def predict(x):
    y = np.dot(x,W) + b
    
    return y

ex_data_01 = np.array([4, 4, 4, 4])    #  4 - 4 + 4 - 4 = 0

print("predicted value = ", predict(ex_data_01) ) 