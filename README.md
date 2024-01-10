# NavierStokes3D.jl

[![Build Status](https://github.com/arsh-k/NavierStokes3D.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/arsh-k/NavierStokes3D.jl/actions/workflows/CI.yml?query=branch%3Aarthur)

NavierStokes3D.jl is a demo developed to simulate flow around a stationary sphere (or cylinder) within a 3D staggered grid domain at different Reynold's numbers. 

### Physical Model (Partial Differential Equations)

The following system of PDEs represent the incompressible Navier-Stokes equations. 

$$
\rho[\frac{\partial \mathbf{V}}{\partial t}+(\mathbf{V} \cdot \nabla \mathbf{V})] =-\nabla p+\mu \nabla^2 \mathbf{V}
$$

$$
\nabla \cdot \mathbf{V} =0
$$

where $t$ is time, $\textbf{V}(\textbf{x},t) = [u,v,w]^T$ is the velocity vector, $p(\textbf{x}, t)$ is the pressure field, $\textbf{x} \in \Omega$ is the spatial coordinate, $\rho$ is the density of the fluid and $\mu$ is the dynamic viscosity of the fluid.

A crucial component for visualizing the vortices around the stationary body is the vorticity. The vorticity components are evaluated as follows:

$$ \omega_x=\frac{\partial w}{\partial y}-\frac{\partial v}{\partial z}$$

$$ \omega_y=-(\frac{\partial u}{\partial z}-\frac{\partial w}{\partial x})$$

$$ \omega_z=\frac{\partial v}{\partial x}-\frac{\partial u}{\partial y}$$

For visualization purposes, we evaluate the magnitude of the vorticity:

$$ 
    |\omega| = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}
$$

### Boundary Conditions

The `set_sphere!` function is used to set a no-slip boundary condition on a sphere of radius `0.05` centered at the coordinates $(x, y, z) = (0.0, 0.0, -0.4)$ (Additionally, it also sets every grid-point within the sphere to a zero velocity).

### Numerical Method

In order to solve the Navier Stokes PDEs, we implement the Chorin's projection method (operator splitting approach). This method involves the splitting of the velocity updates into separate steps based on the physical components of the Navier-Stokes equation.

First, we perform an intermediate velocity update using the gravitational and viscous terms of the N-S equations via a simple Explicit Euler timestepping
scheme (In our code, we have utilized a stress-tensor implementation).

$$
\frac{\mathbf{V}^*-\mathbf{V}^n}{\Delta t}= \mu \nabla^2 \mathbf{V}^n - \rho \mathbf{g}
$$

Then we proceed to evaluate the pressure using a pseudo-transient solver.

Finally, a semi-Lagrangian approach which helps in streamline backtracking of the velocity. 

### Results 

The simulations have been performed to study the flow around a sphere in a three dimensional domain for a Reynolds number $Re=1e6$. The results for two simulations are presented within the folder `docs`. One simulation has been run on a single GPU and the other on 8 GPU nodes.

#### Single GPU simulation 
The lowest resolution has been performed on a single GPU with a grid size of `127*127*255` grid points and `nt=2000` time steps. 

A visualization of the final results of the above mentioned simulation is offered in the following. First, the evolution of vorticity inside the domain is shown via a 3D animation: 

![anim.gif](docs/gpu-solver-sphere/3D_vorticity.gif)

The 3D visualizations for pressure and velocity do not offer much insight. In order to show the evolution of pressure and velocity fields, a longitundinal cross-section through the domain along the $(xz)$ plane is considered. The evolution of pressure, velocity and vorticity fields are depicted in the following figures: 

![anim.gif](docs/gpu-solver-sphere/slice_pressure.gif)

![anim.gif](docs/gpu-solver-sphere/slice_velocity_magnitude.gif)

![anim.gif](docs/gpu-solver-sphere/slice_vorticity.gif)


#### Multiple XPU simulation

The simulation with the higher resolution (in the folder `multi-gpu-solver-sphere`) has been performed on 8 GPU nodes, for a global grid size of `252*252*508` grid points and `nt=2000` time steps. 

A visualization of the final results of the above mentioned simulation is offered in the following. First, the evolution of vorticity inside the domain is shown via a 3D animation: 

![anim.gif](docs/multi-gpu-solver-sphere/3D_vorticity.gif)

The 3D visualizations for pressure and velocity do not offer much insight. In order to show the evolution of pressure and velocity fields, a longitundinal cross-section through the domain along the $(xz)$ plane is considered. The evolution of pressure, velocity and vorticity fields are depicted in the following figures: 

![anim.gif](docs/multi-gpu-solver-sphere/slice_pressure.gif)

![anim.gif](docs/multi-gpu-solver-sphere/slice_velocity_magnitude.gif)

![anim.gif](docs/multi-gpu-solver-sphere/slice_vorticity.gif)

### Weak Scaling

The figure below indicates the normalized execution time as a function of the number of processes. It is observed that even as the global domain size increases as the number of processes increase there is a minor increase in execution time compared to a single process. The minor increase in execution time is due to communication between the multiple GPU nodes via `update_halo!` macro. 

![Weak Scaling](./docs/weak_scaling_navier_stokes_3d_multixpu.png)

To visualize the implications of weak scaling, we had to keep a fixed value of `nt` and `niter` within our weak scaling implementation. This ensures that the number of kernel calls are same for all multi-gpu configurations and remain unaffected by the time step size. The weak scaling implementation is provided in `navier_stokes_3d_multixpu_weak_scaling.jl` and the batch script for the same is provided in `navier_stokes_weak_scaling.sh`.

### Running the software
**NOTE**: To run any simulation on a CPU process, one has to change the variable `USE_GPU` to `false` in the Julia scripts.

The simulation scripts are provided in the `scripts` folder. To launch a simulation locally on CPU processes, the following procedure can be followed (creating a subfolder for saving the outputs at provided stamps): 

```
cd scripts
mkdir out_vis_all
julia --project
include("navier_stokes_3d_multixpu_sphere.jl")
```

Otherwise, a GPU process can also be launched on a single GPU node: 

```
cd scripts
mkdir out_vis_all
sbatch navier_stokes_xpu.sh
```

If the simulation wants to be launched on multiple GPU nodes, a similar procedure can be applied in order to run the baseline simulation on 8 GPUs for 2000 timesteps: 

```
cd scripts
mkdir out_vis_all
sbatch navier_stokes_multixpu.sh
```

Finally, if one wishes to check if our simulation , the procedure would be (activating the testing environment): 

```
cd test
julia
import Pkg
Pkg.activate('.')
Pkg.instantiate()
include("runtests.jl")
``` 