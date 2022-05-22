# Design document
## Physical simulation
This part is mainly implemented in `src/pbf3d.py`.

The goal of this section is to implement the algorithm for position based fluids [Macklin et al. 2013]. We compute the positions and velocities of particle and store them in `positions = taichi.Vector.field(dim, float)` and `velocities = taichi.Vector.field(dim, float)`.
## Scene building
This part is mainly implemented in `src/scene.py`.

The goal of this section is to build an efficient representation of the scene objects from the boundary to obstacles and calculate their influence on the particles. The Taichi function `confine_position_to_scene(p)` should recalculate the positions of particles (viewed as spheres of radius `particle_radius`) run outside of the boundary or into the obstacles.

 - [ ] `confine_position_to_scene(p)`: recalculate the position of the particle at `p` if needed.
## Rendering
This part is mainly implemented in `src/camera.py`.

The goal of this section is to render a picture based on `positions` and scene data. At present, we just consider showing ng particles as spheres of radius `particle_radius`. Time permitting, we would try some complicated methods such as ellipsoid splatting to construct the surface of the fluid.

 - [ ] `render(gui)`.

## Global parameters
 - `screen_res`: a 2D vector, screen resolution
 - `boundary`: a 3D vector, world boundary
 - `dim`: 3, dimension
 - `particle_radius`: a float number, radius of particle in world space
 - `time_delta`: a float number, time step
