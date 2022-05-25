import taichi as ti

from pbf3d import ParticleSystem
from renderer import Renderer

# ti.init(arch=ti.gpu)

arch = ti.cuda if ti._lib.core.with_cuda() else ti.vulkan
ti.init(arch=arch)

screen_res = (1920, 1080)

if __name__ == '__main__':
    ps = ParticleSystem(16)
    rd = Renderer(ps)
    rd.render()
