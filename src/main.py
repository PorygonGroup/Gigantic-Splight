import taichi as ti

from pbf3d import init_particles, move_models, run_pbf
from camera import render 

ti.init(arch = ti.gpu)

screen_res = (1920, 1080)

def main():
    init_particles()
    gui = ti.GUI('PBF3D', screen_res)
    while gui.running and not gui.get_event(gui.ESCAPE):
        move_models()
        run_pbf()
        render(gui)

if __name__ == '__main__':
    main()
