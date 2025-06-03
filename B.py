import numpy as np
from scipy.integrate import solve_ivp, quad

# 微分方程 y'' = -(x+1)y' + 2y + (1-x^2)e^(-x)
#         y'' + (x+1)y' - 2y = (1-x^2)e^(-x)
# 若 y'' = f(x, y, y') :
def ode_f_form(x, y_vec):
    # y_vec = [y, y']
    y, yp = y_vec
    ypp = -(x + 1) * yp + 2 * y + (1 - x**2) * np.exp(-x)
    return [yp, ypp]

# 若 y'' = p(x)y' + q(x)y + r(x)  
# p(x) = -(x + 1)
def p_func(x):
    return -(x + 1)

# q(x) = 2  
def q_func(x):
    return 2

# r(x) = (1-x^2)e^(-x)
def r_func(x):
    return (1 - x**2) * np.exp(-x)

# BC
x_a, x_b = 0.0, 1.0
y_a, y_b = 1.0, 2.0
h = 0.1 

x_eval = np.arange(x_a, x_b + h, h)
x_fine_eval = np.linspace(x_a, x_b, 101) 

print(f"求解微分方程: y'' = -(x+1)y' + 2y + (1-x^2)e^(-x)")
print(f"邊界條件: y({x_a})={y_a}, y({x_b})={y_b}")
print(f"步長 h = {h}\n")


# 方程: (95 - 5*xi)y_{i-1} - 202*yi + (105 + 5*xi)y_{i+1} = (1-xi^2)e^(-xi)
# y0 = y_a, y_n+1 = y_b (n+1 = (b-a)/h = 10)
# 未知數 y1, ..., y9 (共9個)

N_fd = int((x_b - x_a) / h) -1 # 內部未知點的數量
# 內部節點 x_coords[i] = x_{i+1}
x_coords_fd = np.array([x_a + (i+1)*h for i in range(N_fd)])

# 構建矩陣 A 和向量 F
A_fd = np.zeros((N_fd, N_fd))
F_fd = np.zeros(N_fd)

# 填充矩陣和向量
# 公式: -(1+0.5*h*p_i)y_{i-1} + (2+h^2*q_i)y_i - (1-0.5*h*p_i)y_{i+1} = -h^2*r_i [cite: 2]
# 整理 (95 - 5x_i)y_{i-1} - 202y_i + (105 + 5x_i)y_{i+1} = (1-x_i^2)e^{-x_i}

for i in range(N_fd):
    xi = x_coords_fd[i]
    
    # 對角線元素係數 for y_i
    A_fd[i, i] = -202.0
    
    # 次對角線元素係數 for y_{i-1}
    if i > 0:
        A_fd[i, i-1] = 95.0 - 5.0 * xi
        
    # 超對角線元素係數 for y_{i+1}
    if i < N_fd - 1:
        A_fd[i, i+1] = 105.0 + 5.0 * xi
        
    # 右端項 F_i
    F_fd[i] = (1 - xi**2) * np.exp(-xi)
    
    # BC:
    if i == 0: # 第一個方程
        # (95 - 5*x1)*y0 - 202*y1 + (105 + 5*x1)*y2 = (1-x1^2)e^(-x1)
        # -202*y1 + (105 + 5*x1)*y2 = (1-x1^2)e^(-x1) - (95 - 5*x1)*y_a
        F_fd[i] -= (95.0 - 5.0 * xi) * y_a
    if i == N_fd - 1: # 最後一個方程
        # (95 - 5*x_N)*y_{N-1} - 202*y_N + (105 + 5*x_N)*y_{N+1} = (1-x_N^2)e^(-x_N)
        # (95 - 5*x_N)*y_{N-1} - 202*y_N = (1-x_N^2)e^(-x_N) - (105 + 5*x_N)*y_b
        F_fd[i] -= (105.0 + 5.0 * xi) * y_b

# 求解線性系統 {A}{Y} = {F} 
y_internal_fd = np.linalg.solve(A_fd, F_fd)

# 組合完整解（包括邊界點）
y_fd = np.concatenate(([y_a], y_internal_fd, [y_b]))

print("Solution byFinite-Difference method:")
for i, x_val in enumerate(x_eval):
    print(f"y({x_val:.1f}) = {y_fd[i]:.6f}")
print("-" * 30 + "\n")
