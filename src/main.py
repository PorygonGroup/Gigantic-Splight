import taichi as ti

from scene import Scene
from pbf3d import ParticleSystem, Simulator
from renderer import Renderer

# ti.init(arch=ti.gpu)

arch = ti.cuda if ti._lib.core.with_cuda() else ti.vulkan
ti.init(arch=arch)

screen_res = (1920, 1080)

if __name__ == '__main__':
    scene = Scene()
    ps = ParticleSystem(16, 0.03, scene)
    sim = Simulator(ps)
    rd = Renderer(ps, scene)
    while True:
        scene.update()
        sim.step()
        rd.render()
