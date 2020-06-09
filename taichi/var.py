import taichi as ti

a = ti.var(ti.i64)
b = ti.var(ti.i32, shape=1)
c = ti.var(ti.f32, shape=2)
d = ti.var(ti.f32, shape=())
e = ti.var(ti.i64, shape=())

ti.root.dense(ti.ij, (1,1)).place(a)
a[0]=1

print(a)