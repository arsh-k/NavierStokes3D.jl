const USE_GPU = true
using ParallelStencil
using ImplicitGlobalGrid
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using LinearAlgebra, Printf, MPI
using Plots, MAT

"""
    max_g(array)

Returns the maximum of an array among all processes.
"""
max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

"""
    save_array(file_name, array)

Saves an array variable `array` to the file titled `file_name.bin`.
"""
function save_array(Aname,A)
    fname = string(Aname, ".bin")
    out = open(fname, "w"); write(out, A); close(out)
end

"""
    avx(array)

Averages the array in x-direction.
"""
@views avx(A) = 0.5 .*(A[1:end-1,:,:] .+ A[2:end,:,:])

"""
    avy(array)

Averages the array in y-direction.
"""
@views avy(A) = 0.5 .*(A[:,1:end-1,:] .+ A[:,2:end,:])

"""
    avz(array)

Averages the array in z-direction.
"""
@views avz(A) = 0.5 .*(A[:,:,1:end-1] .+ A[:,:,2:end])

"""
    Navier_Stokes_3d_solver_sphere_multixpu(; do_save)

Iterative solver for the incompressible Navier-Stokes equation (flow across a sphere).

The keyword argument includes:
`do_save` : Saves three-dimensional arrays of the desired quantities at checkpoints.
"""
@views function navier_stokes_3d_solver_sphere_multixpu(; do_save=true)
    # physics
    ## dimensionally independent
    lz        = 1.0 # [m]
    ρ         = 1.0 # [kg/m^3]
    vin       = 1.0 # [m/s]
    ## scale
    psc       = ρ*vin^2
    ## nondimensional parameters
    Re        = 1e6    # rho*vsc*ly/μ
    Fr        = Inf   # vsc/sqrt(g*ly)
    lx_lz     = 0.5    # lx/ly
    ly_lz     = 0.5
    r_lz      = 0.05
    ox_lz     = 0.0
    oy_lz     = 0.0
    oz_lz     = -0.4
    ## dimensionally dependent
    lx        = lx_lz*lz
    ly        = ly_lz*lz
    ox        = ox_lz*lz
    oy        = oy_lz*lz
    oz        = oz_lz*lz
    r2         = (r_lz*lz)^2
    μ         = 1/Re*ρ*vin*lz
    g         = 1/Fr^2*vin^2/lz

    # numerics
    nz        = 255
    nx        = floor(Int,nz*lx_lz) #127
    ny        = floor(Int,nz*ly_lz) #127
    me, dims,nprocs,coords  = init_global_grid(nx, ny, nz)
    coords    = Data.Array(coords)
    εit       = 1e-4
    niter     = 50*nx_g()
    nchk      = 1*(nx_g()-1)
    nt        = 2000
    nsave     = 100
    CFLτ      = 0.9/sqrt(3)
    CFL_visc  = 1/5.1
    CFL_adv   = 1.0
    dx,dy,dz  = lx/nx_g(),ly/ny_g(),lz/nz_g()
    dt        = min(CFL_visc*dz^2*ρ/μ,CFL_adv*dz/vin)
    damp      = 2/nz_g()
    dτ        = CFLτ*dz
    # Allocations
    Pr        = @zeros(nx  ,ny  ,nz  )
    dPrdτ     = @zeros(nx-2,ny-2,nz-2)
    C         = @zeros(nx  ,ny  ,nz  )
    C_o       = @zeros(nx  ,ny  ,nz  )
    τxx       = @zeros(nx  ,ny  ,nz  )
    τyy       = @zeros(nx  ,ny  ,nz  )
    τzz       = @zeros(nx  ,ny  ,nz  )
    τxy       = @zeros(nx-1,ny-1,nz-2  )
    τxz       = @zeros(nx-1,ny-2  ,nz-1)
    τyz       = @zeros(nx-2  ,ny-1,nz-1)
    Vx        = @zeros(nx+1,ny,nz)
    Vy        = @zeros(nx  ,ny+1,nz)
    Vz        = @zeros(nx  ,ny, nz+1)
    Vx_o      = @zeros(nx+1,ny ,nz)
    Vy_o      = @zeros(nx  ,ny+1, nz)
    Vz_o      = @zeros(nx  ,ny, nz+1)
    ∇V        = @zeros(nx  ,ny  ,nz )
    Rp        = @zeros(nx-2,ny-2,nz-2)
    x = LinRange(0.5dx, lx-0.5dx,nx)
    y = LinRange(0.5dy, ly-0.5dy,ny)
    # Initial conditions
    Vprof = @zeros(nx,ny)
    Vprof .= Data.Array([4*vin*(x_g(ix,dx,Vprof)+0.5*dx)/lx*(1.0-(x_g(ix,dx,Vprof)+0.5dx)/lx) + 4*vin*(y_g(iy,dy,Vprof)+0.5dy)/ly*(1.0-(y_g(iy,dy,Vprof)+0.5dy)/ly) for  ix = 1:size(Vprof, 1), iy = 1:size(Vprof, 2) ]) 
    Vz[:,:,1] .= Vprof
    update_halo!(Vz)
    Pr  = Data.Array([-z_g(iz,dz,Pr)*ρ*g   for ix=1:size(Pr,1),iy=1:size(Pr,2),iz=1:size(Pr,3)])
    update_halo!(Pr)
    
    if do_save
        nx_v, ny_v, nz_v = (nx - 2) * dims[1], (ny - 2) * dims[2], (nz - 2) * dims[3]
        (nx_v * ny_v * nz_v * sizeof(Data.Number) > 0.8 * Sys.free_memory()) && error("Not enough memory for visualization.")
        Pr_v   = zeros(nx_v, ny_v, nz_v) 
        Vx_v   = zeros(nx_v, ny_v, nz_v) 
        Vy_v   = zeros(nx_v, ny_v, nz_v) 
        Vz_v   = zeros(nx_v, ny_v, nz_v) 
        Pr_inn = zeros(nx - 2, ny - 2, nz - 2) 
        Vx_inn = zeros(nx - 2, ny - 2, nz - 2) 
        Vy_inn = zeros(nx - 2, ny - 2, nz - 2) 
        Vz_inn = zeros(nx - 2, ny - 2, nz - 2) 
    end

    for it = 1:nt
        err_evo = Float64[]; iter_evo = Float64[]
        @parallel update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)
        @parallel predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)
        @parallel set_sphere_multixpu!(C,Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2,nx,ny,nz,coords)
        update_halo!(Vx,Vy,Vz)
        @parallel update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)
        if me==0
            println("#it = $it")
        end
        for iter = 1:niter
            @parallel update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)
            @parallel update_Pr!(Pr,dPrdτ,dτ)
            set_bc_Pr!(Pr, 0.0)
            update_halo!(Pr)
            if iter % nchk == 0
                @parallel compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)
                err = max_g(abs.(Rp))*lz^2/psc
                push!(err_evo, err); push!(iter_evo,iter/nz)
                if me==0
                    @printf("  #iter = %d, err = %1.3e\n", iter, err)
                end
                if err < εit || !isfinite(err) break end
            end
        end
        @parallel correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)
        @parallel set_sphere_multixpu!(C,Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2,nx,ny,nz,coords)
        set_bc_Vel!(Vx, Vy, Vz, Vprof)
        update_halo!(Vx,Vy,Vz)
        Vx_o .= Vx; Vy_o .= Vy; Vz_o .= Vz; C_o .= C
        @parallel advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,C,C_o,dt,dx,dy,dz)
        update_halo!(Vx,Vy,Vz)
        # Saving the arrays
        if do_save && it % nsave == 0
            Pr_inn .= Array(Pr[2:end-1, 2:end-1, 2:end-1]); gather!(Pr_inn, Pr_v)
            Vx_inn .= Array(avx(Vx)[2:end-1, 2:end-1, 2:end-1]); gather!(Vx_inn, Vx_v)
            Vy_inn .= Array(avy(Vy)[2:end-1, 2:end-1, 2:end-1]); gather!(Vy_inn, Vy_v)
            Vz_inn .= Array(avz(Vz)[2:end-1, 2:end-1, 2:end-1]); gather!(Vz_inn, Vz_v)
        end

        if do_save && it % nsave == 0 && me==0
            println("Saving")
            save_array("./out_vis_all/out_Vx_$it", convert.(Float32, Array(Vx_v)))
            save_array("./out_vis_all/out_Vy_$it", convert.(Float32, Array(Vy_v)))
            save_array("./out_vis_all/out_Vz_$it", convert.(Float32, Array(Vz_v)))
            save_array("./out_vis_all/out_Pr_$it", convert.(Float32, Array(Pr_v)))
        end
    end
    finalize_global_grid()
    return
end

"""
    ∇V() esc(:( @d_xa(Vx)/dx + @d_ya(Vy)/dy +@d_za(Vz)/dz))

Computes the divergence of the velocity field.
"""
macro ∇V() esc(:( @d_xa(Vx)/dx + @d_ya(Vy)/dy +@d_za(Vz)/dz)) end


"""
    update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)

Computes the shear stress tensor.
"""
@parallel function update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)
    @all(τxx) = 2μ*(@d_xa(Vx)/dx - @∇V()/3.0)  
    @all(τyy) = 2μ*(@d_ya(Vy)/dy - @∇V()/3.0)
    @all(τxy) =  μ*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    @all(τzz) = 2μ*(@d_za(Vz)/dz - @∇V()/3.0)
    @all(τxz) =  μ*(@d_zi(Vx)/dz + @d_xi(Vz)/dx)
    @all(τyz) =  μ*(@d_zi(Vy)/dz + @d_yi(Vz)/dy)

    return
end

"""
    predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)

Computes first intermediate velocity term with viscous and gravitational terms of the NS equations (Chorin's projection methodology).
"""
@parallel function predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)
    @inn(Vx) = @inn(Vx) + dt/ρ*(@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz)
    @inn(Vy) = @inn(Vy) + dt/ρ*(@d_yi(τyy)/dy + @d_xa(τxy)/dx  + @d_za(τyz)/dz)
    @inn(Vz) = @inn(Vz) + dt/ρ*(@d_zi(τzz)/dz + @d_xa(τxz)/dx  + @d_ya(τyz)/dy - ρ*g)
    return
end

"""
    update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)

Computes the divergence of the velocity using the divergence computation macro.
"""
@parallel function update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)
    @all(∇V) = @∇V()
    return
end

"""
    update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)

Computes the pressure gradient with respect to the pseudo-time. The `damp` constant determines the rate at which the 
pseudo-transient term tends to zero as we approach steady state.
"""
@parallel function update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)
    @all(dPrdτ) = @all(dPrdτ)*(1.0-damp) + dτ*(@d2_xi(Pr)/dx/dx + @d2_yi(Pr)/dy/dy + @d2_zi(Pr)/dz/dz - ρ/dt*@inn(∇V))
    return
end

"""
    update_Pr!(Pr,dPrdτ,dτ)

Updates the pressure field using pressure gradient with respect to pseudo-time.
"""
@parallel function update_Pr!(Pr,dPrdτ,dτ)
    @inn(Pr) = @inn(Pr) + dτ*@all(dPrdτ)
    return
end

"""
    compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)

Computes the residual of the PDE required to attain the pressure in the current time step. 
"""
@parallel function compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)
    @all(Rp) = @d2_xi(Pr)/dx/dx + @d2_yi(Pr)/dy/dy + @d2_zi(Pr)/dz/dz - ρ/dt*@inn(∇V)
    return
end

"""
    correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)

Computes second intermediate velocity term with spatial pressure-gradients.
"""
@parallel function correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)
    @inn(Vx) = @inn(Vx) - dt/ρ*@d_xi(Pr)/dx
    @inn(Vy) = @inn(Vy) - dt/ρ*@d_yi(Pr)/dy
    @inn(Vz) = @inn(Vz) - dt/ρ*@d_zi(Pr)/dz 
    return
end

"""
    bc_yz!(A)

Sets zero-flux boundary condition in the yz-plane for the quantity `A`.
"""
@parallel_indices (iy, iz) function bc_yz!(A)
    if iy <= size(A,2) && iz <= size(A,3)
        A[1  , iy,  iz] = A[2    , iy,  iz]
        A[end, iy,  iz] = A[end-1, iy,  iz]
    end
    return
end

"""
    bc_xz!(A)

Sets zero-flux boundary condition in the xz-plane for the quantity `A`.
"""
@parallel_indices (ix, iz) function bc_xz!(A)
    if ix <= size(A,1) && iz <= size(A,3)
        A[ix, 1  ,  iz] = A[ix,     2, iz]
        A[ix, end,  iz] = A[ix, end-1, iz]
    end
    return
end

"""
    bc_xy!(A)

Sets zero-flux boundary condition in the xy-plane for the quantity `A`.
"""
@parallel_indices (ix, iy) function bc_xy!(A)
    if ix <= size(A,1) && iy <= size(A,2)
        A[ix, iy,    1] = A[ix,  iy,     2]
        A[ix, iy,  end] = A[ix,  iy, end-1]
    end
    return
end

"""
    bc_zV!(A, V)

Sets zero-flux boundary condition in one xy-plane and a Dirichlet boundary condition = `V` in the 
other xy-plane (index `1`) for the quantity `A`. 
"""
@parallel_indices (ix, iy) function bc_zV!(A, V)
    A[ix,   iy,   1] = V[    ix,    iy]
    A[ix,   iy, end] = A[ix, iy, end-1]
    return
end

"""
    bc_xyval!(A, V)

Sets zero-flux boundary condition in one xy-plane (index `1`) and a Dirichlet boundary condition = `val` in the other 
xy-plane (index `end-1`) for the quantity `A`.
"""
@parallel_indices (ix, iy) function bc_xyval!(A, val)
    A[ix, iy,   1] = A[ix, iy,  2]
    A[ix, iy, end] = val
    return
end

"""
    set_bc_Vel!(Vx, Vy, Vz, Vprof)

Sets the velocity component (`Vx`, `Vy`, `Vz`) boundary conditions on the entire domain. `Vprof` is provided as
initial condition for the `Vz` velocity component within the xy-plane.
"""
function set_bc_Vel!(Vx, Vy, Vz, Vprof)
    @parallel bc_xy!(Vx)
    @parallel bc_xz!(Vx)
    @parallel bc_xy!(Vy)
    @parallel bc_yz!(Vy)
    @parallel bc_xz!(Vz)
    @parallel bc_yz!(Vz)
    @parallel bc_zV!(Vz, Vprof)
    return
end

"""
    set_bc_Pr!(Pr, val)

Sets the pressure boundary conditions on the entire domain.
"""
function set_bc_Pr!(Pr, val)
    @parallel bc_xz!(Pr)
    @parallel bc_yz!(Pr)
    @parallel bc_xyval!(Pr, val)
    return
end


"""
    backtrack!(A,A_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)

Performs backtracking for fluid velocities in the advection scheme.
"""
@inline function backtrack!(A,A_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    δx,δy,δz    = dt*vxc/dx, dt*vyc/dy , dt*vzc/dz
    ix1      = clamp(floor(Int,ix-δx),1,size(A,1))
    iy1      = clamp(floor(Int,iy-δy),1,size(A,2))
    iz1      = clamp(floor(Int,iz-δz),1,size(A,3))
    ix2,iy2,iz2  = clamp(ix1+1,1,size(A,1)),clamp(iy1+1,1,size(A,2)),clamp(iz1+1,1,size(A,3))
    δx = (δx>0) - (δx%1); δy = (δy>0) - (δy%1) ; δz = (δz>0) - (δz%1)
    fx11      = lerp(A_o[ix1,iy1,iz1],A_o[ix2,iy1,iz1],δx)
    fx12      = lerp(A_o[ix1,iy2,iz1],A_o[ix2,iy2,iz1],δx)
    fx1 = lerp(fx11,fx12,δy)

    fx21      = lerp(A_o[ix1,iy1,iz2],A_o[ix2,iy1,iz2],δx)
    fx22      = lerp(A_o[ix1,iy2,iz2],A_o[ix2,iy2,iz2],δx)
    fx2 = lerp(fx21,fx22,δy)

    A[ix,iy,iz] = lerp(fx1,fx2,δz)
    return
end

"""
    lerp(a,b,t)

Performs linear interpolation of the form `b*t + a*(1-t)`.
"""
@inline lerp(a,b,t) = b*t + a*(1-t)

"""
    advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,dt,dx,dy,dz)

Implements semi-Lagrangian advection scheme with linear interpolation. 
"""
@parallel_indices (ix,iy,iz) function advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,C,C_o,dt,dx,dy,dz)
    if ix > 1 && ix < size(Vx,1) && iy <= size(Vx,2) && iz<=size(Vx,3)
        vxc      = Vx_o[ix,iy,iz]
        vyc      = 0.25*(Vy_o[ix-1,iy,iz]+Vy_o[ix-1,iy+1,iz]+Vy_o[ix,iy,iz]+Vy_o[ix,iy+1,iz])
        vzc      = 0.25*(Vz_o[ix-1,iy,iz]+Vz_o[ix-1,iy,iz+1]+Vz_o[ix,iy,iz]+Vz_o[ix,iy,iz+1])
        backtrack!(Vx,Vx_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    if iy > 1 && iy < size(Vy,2) && ix <= size(Vy,1) && iz<=size(Vy,3)
        vxc      = 0.25*(Vx_o[ix,iy-1,iz]+Vx_o[ix+1,iy-1,iz]+Vx_o[ix,iy,iz]+Vx_o[ix+1,iy,iz])
        vyc      = Vy_o[ix,iy,iz]
        vzc      = 0.25*(Vz_o[ix,iy-1,iz]+Vz_o[ix,iy-1,iz+1]+Vz_o[ix,iy,iz]+Vz_o[ix,iy,iz+1])
        backtrack!(Vy,Vy_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end

    if iz > 1 && iz < size(Vz,3) && ix <= size(Vz,1) && iy<=size(Vz,2)
        vxc      = 0.25*(Vx_o[ix,iy,iz-1]+Vx_o[ix+1,iy,iz-1]+Vx_o[ix,iy,iz]+Vx_o[ix+1,iy,iz])
        vyc      = 0.25*(Vy_o[ix,iy,iz-1]+Vy_o[ix,iy+1,iz-1]+Vy_o[ix,iy,iz]+Vy_o[ix,iy+1,iz])
        vzc      = Vz_o[ix,iy,iz]
        backtrack!(Vz,Vz_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    return
end


"""
    set_sphere_multixpu!(Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2)

Sets a spherical object in the three-dimensional domain with a no-slip surface. (For multiple processes, it is specialized to be only set in a single process.)
"""
@parallel_indices (ix,iy,iz) function set_sphere_multixpu!(C,Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2,nx,ny,nz,coords)
    xc,yc,zc = (coords[1]*(nx-2)+(ix-1))*dx+dx/2-lx/2,(coords[2]*(ny-2)+(iy-1))*dy+dy/2-ly/2,(coords[3]*(nz-2)+(iz-1))*dz+dz/2-lz/2
    xv,yv,zv = (coords[1]*(nx-2)+(ix-1))*dx-lx/2,(coords[2]*(ny-2)+(iy-1))*dy-ly/2,(coords[3]*(nz-2)+(iz-1))*dz-lz/2
    
    if checkbounds(Bool,Vx,ix,iy,iz)
        xr = (xv-ox)
        yr = (yc-oy)
        zr = (zc-oz)
        if xr*xr/r2 + yr*yr/r2 + zr*zr/r2 < 1.0
            Vx[ix,iy,iz] = 0.0
        end
    end
    if checkbounds(Bool,Vy,ix,iy,iz)
        xr = (xc-ox)
        yr = (yv-oy)
        zr = (zc-oz)
        if  xr*xr/r2 + yr*yr/r2 + zr*zr/r2 < 1.0
            Vy[ix,iy,iz] = 0.0
        end
    end
    if checkbounds(Bool,Vz,ix,iy,iz)
        xr = (xc-ox)
        yr = (yc-oy)
        zr = (zv-oz)
        if xr*xr/r2 + yr*yr/r2 + zr*zr/r2 < 1.0
            Vz[ix,iy,iz] = 0.0
        end
    end
    return
end

navier_stokes_3d_solver_sphere_multixpu()
