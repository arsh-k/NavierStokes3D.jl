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

The `set_sphere!` function is used to set a no-slip boundary condition on a sphere of radius `0.05` centered at the coordinates $(x, y, z) = (0.0, 0.0, -0.4)$. 

### Numerical Method

In order to solve the Navier Stokes PDEs, we implement the Chorin's projection method (operator splitting approach). This method involves the splitting of the velocity updates
into separate steps based on the components of the Navier-Stokes equation.

First, we perform an intermediate velocity update using the gravitational and viscous terms of the N-S equations. 

Then we proceed to evaluate the pressure using a pseudo-transient implementation. 

Finally, a semi-Lagrangian approach which helps in streamline backtracking of the velocity. 

### Weak Scaling

The figure below indicates the normalized execution time as a function of the number of processes. It is observed that even as the global domain size increases as the number of processes increase there is a minor increase in execution time compared to a single process. The minor increase in execution time is due to communication between the multiple GPU nodes via `update_halo!` macro. 

![Weak Scaling](./docs/weak_scaling_navier_stokes_3d_multixpu.png)

To visualize the implications of weak scaling, we had to keep a fixed value of `nt` and `niter` within our weak scaling implementation. This ensures that the number of kernel calls are same for all multi-gpu configurations and remain unaffected by the time step size. The weak scaling implementation is provided in `navier_stokes_3d_multixpu_weak_scaling.jl` and the batch script for the same is provided in `navier_stokes_weak_scaling.sh`.

### Results 

### Discussion and Conclusion