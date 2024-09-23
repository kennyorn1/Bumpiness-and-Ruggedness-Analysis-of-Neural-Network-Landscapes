from sympy import hessian
import sympy as sym
import numpy as np

# 1 特征值： [ 2. -2.] trace: 0.0 rugg: 4.0 bump: 0.0
# x1, x2 = sym.symbols("x1 x2")
# f3 = x1**2 - x2**2
# a = hessian(f3, (x1, x2))

# # print(a.subs([(x1, 0), (x2, 0)]))
# mat = np.array([[2, 0], [0, -2]])
# eigenvalue, featurevector = np.linalg.eig(mat)

# print(
#     "特征值：",
#     eigenvalue,
#     "trace:",
#     sum(eigenvalue),
#     "rugg:",
#     sum(np.abs(eigenvalue)),
#     "bump:",
#     sum(eigenvalue) / sum(np.abs(eigenvalue)),
# )


# 2 特征值： [0. 0.] trace: 0.0 rugg: 0.0 bump: nan
# x1, x2 = sym.symbols("x1 x2")
# f3 = x1**6 - x2**6
# a = hessian(f3, (x1, x2))

# # print(a.subs([(x1, 0), (x2, 0)]))
# mat = np.array([[0, 0], [0, 0]])
# eigenvalue, featurevector = np.linalg.eig(mat)

# print(
#     "特征值：",
#     eigenvalue,
#     "trace:",
#     sum(eigenvalue),
#     "rugg:",
#     sum(np.abs(eigenvalue)),
#     "bump:",
#     sum(eigenvalue) / sum(np.abs(eigenvalue)),
# )

# 3 特征值： [2. 0.] trace: 2.0 rugg: 2.0 bump: 1.0
x1, x2 = sym.symbols("x1 x2")
f3 = x1**2 - x2**6
a = hessian(f3, (x1, x2))

# print(a.subs([(x1, 0), (x2, 0)]))
mat = np.array([[2, 0], [0, 0]])
eigenvalue, featurevector = np.linalg.eig(mat)

print(
    "特征值：",
    eigenvalue,
    "trace:",
    sum(eigenvalue),
    "rugg:",
    sum(np.abs(eigenvalue)),
    "bump:",
    sum(eigenvalue) / sum(np.abs(eigenvalue)),
)
