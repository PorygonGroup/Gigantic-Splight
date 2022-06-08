import time

import taichi as ti

from scene import Scene
from pbf3d import ParticleSystem, Simulator
import pbf3d
from renderer import Renderer

# ti.init(arch=ti.gpu)

arch = ti.cuda if ti._lib.core.with_cuda() else ti.vulkan
ti.init(arch=arch)

screen_res = (1920, 1080)

if __name__ == '__main__':
    scene = Scene()
    ps = ParticleSystem(pbf3d.particle_num, 0.01, scene)
    sim = Simulator(ps)
    rd = Renderer(ps, scene)
    bit = 10000
    while True:
        scene.update()
        if bit>0:
            sim.step()
            bit -=1
        rd.render()
