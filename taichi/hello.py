import taichi as ti

@ti.kernel
def hello(i: ti.i32):
    a = 40
    print('Hello world', a + i)

hello(2)
