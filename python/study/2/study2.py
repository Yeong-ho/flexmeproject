import numpy as np
from datetime import datetime

np.random.seed(0)

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

# 최종출력은 y = sigmoid(Wx+b) 이며, 손실함수는 cross-entropy 로 나타냄

def sigmoid(z):
    return 1 / (1+np.exp(-z))

x_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(10,1)   
t_data = np.array([0, 0, 0, 0,  0,  0,  1,  1,  1,  1]).reshape(10,1)

print("x_data.shape = ", x_data.shape, ", t_data.shape = ", t_data.shape)

W = np.random.rand(1,1)  
b = np.random.rand(1)  

print("W = ", W, ", W.shape = ", W.shape, ", b = ", b, ", b.shape = ", b.shape)

def loss_func(x, t):
    
    delta = 1e-7    # log 무한대 발산 방지
    
    z = np.dot(x, W) + b
    y = sigmoid(z)
    
    # cross-entropy 
    return  -np.sum( t*np.log(y + delta) + (1-t)*np.log((1 - y)+delta ) ) 

# 손실함수 값 계산 함수
# 입력변수 x, t : numpy type
def error_val(x, t):
    delta = 1e-7    # log 무한대 발산 방지
    
    z = np.dot(x, W) + b
    y = sigmoid(z)
    
    # cross-entropy 
    return  -np.sum( t*np.log(y + delta) + (1-t)*np.log((1 - y)+delta ) ) 

learning_rate = 1e-4  # 발산하는 경우, 1e-3 ~ 1e-6 등으로 바꾸어서 실행

f = lambda x : loss_func(x_data,t_data)  

print("Initial error value = ", error_val(x_data, t_data) )

start_time = datetime.now()

for step in  range(100001):  
    
    W -= learning_rate * numerical_derivative(f, W)
    
    b -= learning_rate * numerical_derivative(f, b)
    
    if (step % 5000 == 0):
        print("step = ", step, "error value = ", error_val(x_data, t_data) )
        
end_time = datetime.now()
        
print("")
print("Elapsed Time => ", end_time - start_time)