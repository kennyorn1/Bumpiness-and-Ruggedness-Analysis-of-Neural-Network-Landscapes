import numpy as np
import matplotlib.pyplot as plt

# 3D图需要额外导入模块
from mpl_toolkits.mplot3d import Axes3D

# 将默认figure图转化为3D图
fig = plt.figure()
ax = Axes3D(fig)

# 给出x，y的坐标数据
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
# 画出网格线
X, Y = np.meshgrid(X, Y)

# 给出高度Z的值
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# 画出图像 rstride:横向的分割线跨度(越小越密集) cstride:纵向的分割线跨度(越小越密集)
ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap("rainbow"), edgecolor="black"
)

# 将颜色进行投影
# zdir后的参数决定从哪个方位进行投影 offset的参数表示投影到该方位坐标的哪个点对应的坐标平面
ax.contourf(X, Y, Z, zdir="z", offset=-2, cmap="rainbow")

# 限制画图的坐标轴范围
ax.set_zlim(-2, 2)

plt.show()
