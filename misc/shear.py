import numpy as np

sh_x = 2.0
sh_y = 2.0
M_shear = np.array([[1,sh_x,0],[sh_y, 1, 0],[0,0,1]])
x_shear = np.array([[1,sh_x,0],[0, 1, 0],[0,0,1]])
y_shear = np.array([[1,0,0],[sh_y, 1, 0],[0,0,1]])

print(x_shear@y_shear)
print("-----------")
print(M_shear)
