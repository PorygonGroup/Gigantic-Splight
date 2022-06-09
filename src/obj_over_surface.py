from pbf3d import ParticleSystem
import pbf3d
import taichi as ti

@ti.data_oriented
class BallObj:
    def __init__(self,pos,radius,color,ps:ParticleSystem):
        self.radius = radius
        self.color = color
        self.ps = ps
        self.pos = ti.Vector.field(3, float, shape=1)
        self.zv = ti.field(float,shape=())
        self.pos[0][0]=pos[0]
        self.pos[0][1]=pos[1]
        self.pos[0][2]=pos[2]

    @ti.kernel
    def update(self):
        x=0.0
        y=0.0
        for p_i in self.ps.p:
            x+=self.ps.p[p_i][0]
            y+=self.ps.p[p_i][1]
        x=x/pbf3d.particle_num
        y=y/pbf3d.particle_num
        z=0.0
        k=0
        for p_i in self.ps.p:
            dx = x - self.ps.p[p_i][0]
            dy = y - self.ps.p[p_i][1]
            if dx*dx+dy*dy < 1:
                z+=self.ps.p[p_i][2]
                k+=1
        z=z/k
        self.pos[0][0]=x
        self.pos[0][1]=y
        self.pos[0][2]=z*2.0*0.1+self.zv[None]*0.9
        self.zv[None] = self.zv[None]*0.8+z*0.2*2