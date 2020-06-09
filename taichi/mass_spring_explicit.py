#-*- utf-8 -*-
import taichi as ti

# 调试模式
# debug=True
ti.init(debug=True)

# 最大的粒子数目
max_num_particles = 256

# 时间增量Δt，调整这个可以看到弹簧的微小的变化
dt = 1e-3

# 当前的粒子数量
num_particles = ti.var(ti.i32, shape=())
# 劲度系数
spring_stiffness = ti.var(ti.f32, shape=())
# 暂停状态
paused = ti.var(ti.i32, shape=())
# 阻尼
damping = ti.var(ti.f32, shape=())

# 粒子质量
particle_mass = 1
# 地面边界
bottom_y = 0.05

# 粒子的位置 (X,Y)坐标
x = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
# 粒子的速度
v = ti.Vector(2, dt=ti.f32, shape=max_num_particles)

# 没有使用
# A = ti.Matrix(2, 2, dt=ti.f32, shape=(max_num_particles, max_num_particles))
# b = ti.Vector(2, dt=ti.f32, shape=max_num_particles)

# 止动长度
# rest_length[i, j] = 0 means i and j are not connected
rest_length = ti.var(ti.f32, shape=(max_num_particles, max_num_particles))

# 碰撞连接半径
connection_radius = 0.15

# （XY）重力加速度
gravity = [0, -9.8]


# 分步
@ti.kernel
def substep():
    # 计算力和新的速度
    # Compute force and new velocity
    n = num_particles[None]
    for i in range(n-1, -1, -1):
        v[i] *= ti.exp(-dt * damping[None])  # damping
        total_force = ti.Vector(gravity) * particle_mass  # 初始受力只有重力
        for j in range(n):
            # 对于互相连接的粒子用胡克定律计算力
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                total_force += -spring_stiffness[None] * (
                    x_ij.norm() -
                    rest_length[i, j]) * x_ij.normalized()  # 加上弹簧之间的力
        v[i] += dt * total_force / particle_mass  # 牛顿第二定律

    # 碰撞地面
    # Collide with ground
    for i in range(n):
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0

    # 计算新的位置
    # Compute new position
    for i in range(num_particles[None]):
        x[i] += v[i] * dt


# 插入一个新的粒子
@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32):
    # Taichi doesn't support using Matrices as kernel arguments yet
    new_particle_id = num_particles[None]  # 下标序号
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1

    # 连接现有的粒子
    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()  # 计算距离
        if dist < connection_radius:
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1


# 窗口大小 512*512
# XY 坐标范围 [0, 1]
# 左下角为（0，0），右上为正方向
gui = ti.GUI('Mass Spring System', res=(512, 512), background_color=0xdddddd)

# 调大劲度系数，弹簧会比较坚硬
# 调小劲度系数。弹簧会比较软
spring_stiffness[None] = 10000

# 调大阻尼，弹簧不容易变速
# 调小阻尼，弹簧容易改变速度
damping[None] = 20

new_particle(0.3, 0.3)
new_particle(0.3, 0.4)
new_particle(0.4, 0.4)

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]
        elif e.key == ti.GUI.LMB:
            new_particle(e.pos[0], e.pos[1])
        elif e.key == 'c':
            num_particles[None] = 0
            rest_length.fill(0)
        elif e.key == 's':
            if gui.is_pressed('Shift'):
                spring_stiffness[None] /= 1.1
            else:
                spring_stiffness[None] *= 1.1
        elif e.key == 'd':
            if gui.is_pressed('Shift'):
                damping[None] /= 1.1
            else:
                damping[None] *= 1.1

    if not paused[None]:
        for step in range(10):
            substep()

    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=5)

    gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)

    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
    gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
    gui.text(content=f'S: Spring stiffness {spring_stiffness[None]:.1f}',
             pos=(0, 0.9),
             color=0x0)
    gui.text(content=f'D: damping {damping[None]:.2f}',
             pos=(0, 0.85),
             color=0x0)
    gui.show()
