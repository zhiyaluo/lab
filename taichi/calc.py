import taichi as ti

@ti.kernel
def calc() -> ti.i32:
    s = 0
    for i in range(10):
        s += i
    return s

v = calc()
print(v)
