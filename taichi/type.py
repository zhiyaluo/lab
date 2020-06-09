import taichi as ti

# ti.init()

x = ti.var(dt=ti.f32, shape=1)
y = ti.Vector(3, dt=ti.f32, shape=2)
z = ti.Matrix([[1.0, 0.0], [0.1, 0.2]])

print("x = ", x)
print("y = ", y)
print("z = ", z)
