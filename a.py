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

# IVP 1: y1'' = p(x)y1' + q(x)y1 + r(x), y1(a)=alpha, y1'(a)=0 
# y1'' = -(x-1)y1' + 2y1 + (1-x^2)e^(-x)
def ivp1_func(x, y_vec):
    # y_vec = [y1, y1']
    y1, y1p = y_vec
    y1pp = p_func(x) * y1p + q_func(x) * y1 + r_func(x)
    return [y1p, y1pp]

# IVP 2: y2'' = p(x)y2' + q(x)y2, y2(a)=0, y2'(a)=1
# y2''= -(x-1)y2' + 2y2
def ivp2_func(x, y_vec):
    # y_vec = [y2, y2']
    y2, y2p = y_vec
    y2pp = p_func(x) * y2p + q_func(x) * y2
    return [y2p, y2pp]

# 初始條件 a=0,b=1
y1_init = [y_a, 0.0]  # [y1(a), y1'(a)]
y2_init = [0.0, 1.0]  # [y2(a), y2'(a)]

sol1 = solve_ivp(ivp1_func, [x_a, x_b], y1_init, dense_output=True, t_eval=x_eval, rtol=1e-8, atol=1e-8)
sol2 = solve_ivp(ivp2_func, [x_a, x_b], y2_init, dense_output=True, t_eval=x_eval, rtol=1e-8, atol=1e-8)

y1_at_b = sol1.y[0, -1] # y1(b)
y2_at_b = sol2.y[0, -1] # y2(b)

c = (y_b - y1_at_b) / y2_at_b 
print(f"計算得常數 c = {c:.6f}\n")
# 計算最終解 y(x) = y1(x) + c*y2(x) 
y_shooting = sol1.y[0] + c * sol2.y[0]

print("Solution by Shooting method:")
for i, x_val in enumerate(x_eval):
    print(f"y({x_val:.1f}) = {y_shooting[i]:.6f}")
print("-" * 30 + "\n")
