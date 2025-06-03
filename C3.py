import numpy as np
from scipy.integrate import quad, solve_bvp

# 微分方程 y'' = -(x+1)y' + 2y + (1-x^2)e^(-x)
#         y'' + (x+1)y' - 2y = (1-x^2)e^(-x)

# (e^(x^2/2 + x)y')' - 2e^(x^2/2 + x)y = (1-x^2)e^(x^2/2)
# 對應 -(P_s y')' + Q_s y = G_s
# P_s(x) = -e^(x^2/2 + x)
# Q_s(x) = -2e^(x^2/2 + x)
# G_s(x) = (1-x^2)e^(x^2/2)

def P_s_func(x):
    return -np.exp(x**2 / 2 + x)

def Q_s_func(x):
    return -2 * np.exp(x**2 / 2 + x)

def G_s_func(x):
    return (1 - x**2) * np.exp(x**2 / 2)

# y1(x) = 1 + x
def y1_func(x):
    return 1 + x

def y1_prime_func(x):
    return 1.0

# F_eff(x) = G_s(x) - [-d/dx(P_s(x)y1') + Q_s(x)y1(x)]
# -d/dx(P_s(x)y1') = (x+1)e3(x^2/2 + x)
# Q_s(x)y1(x) = -2e3(x^2/2 + x)(1+x)
# So, [-d/dx(P_s(x)y1') + Q_s(x)y1(x)] = -(x+1)e3(x^2/2+x)
def F_eff_func(x):
    term_from_y1 = -(x + 1) * np.exp(x**2 / 2 + x)
    return G_s_func(x) - term_from_y1

# BC
x_a, x_b = 0.0, 1.0
y_a, y_b = 1.0, 2.0
h = 0.1

print(f"求解微分方程: y'' = -(x+1)y' + 2y + (1-x^2)e^(-x)")
print(f"邊界條件: y({x_a})={y_a}, y({x_b})={y_b}")

# φ_k(0) = 0, φ_k(1) = 0
phi_funcs = [
    lambda x: x * (1 - x),        # φ_1(x) = x - x^2
    lambda x: x**2 * (1 - x),     # φ_2(x) = x^2 - x^3
    lambda x: x**3 * (1 - x)      # φ_3(x) = x^3 - x^4
]

phi_prime_funcs = [
    lambda x: 1 - 2 * x,          # φ_1'(x)
    lambda x: 2 * x - 3 * x**2,   # φ_2'(x)
    lambda x: 3 * x**2 - 4 * x**3 # φ_3'(x)
]

n_ritz = 3
A = np.zeros((n_ritz, n_ritz))
b_vec = np.zeros(n_ritz)

# 計算矩陣 A 和向量 b
# A_ij = ∫ [ P_s(x)φ_i'(x)φ_j'(x) + Q_s(x)φ_i(x)φ_j(x) ] dx
# b_i = ∫ F_eff(x)φ_i(x) dx
for i in range(n_ritz):
    # 計算 b_i
    integrand_b = lambda x: F_eff_func(x) * phi_funcs[i](x)
    b_vec[i], _ = quad(integrand_b, x_a, x_b)

    for j in range(n_ritz):
        integrand_A = lambda x: (P_s_func(x) * phi_prime_funcs[i](x) * phi_prime_funcs[j](x) +
                                 Q_s_func(x) * phi_funcs[i](x) * phi_funcs[j](x))
        A[i, j], _ = quad(integrand_A, x_a, x_b)

c_coeffs = np.linalg.solve(A, b_vec)
print("\n計算得係數 c:", c_coeffs)

#Solution
def y_approx_ritz(x):
    y2_val = sum(c_coeffs[k] * phi_funcs[k](x) for k in range(n_ritz))
    return y1_func(x) + y2_val

x_eval = np.arange(x_a, x_b + h, h)
y_ritz_eval = np.array([y_approx_ritz(val) for val in x_eval])

print("\nSolution by Variation approach (n=3):")
for xi, yi in zip(x_eval, y_ritz_eval):
    print(f"y({xi:.1f}) = {yi:.6f}")
