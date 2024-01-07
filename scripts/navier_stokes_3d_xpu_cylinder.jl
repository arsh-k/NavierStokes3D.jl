const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using LinearAlgebra, Printf
using MAT, Plots

"""
    save_array(file_name, array)

Saves an array variable `array` to the file titled `file_name.bin`.
"""
function save_array(Aname,A)
    fname = string(Aname, ".bin")
    out = open(fname, "w"); write(out, A); close(out)
end

"""
    navier_stokes_3D_solver_cylinder_xpu(; do_vis, do_save)

Iterative solver for the incompressible Navier-Stokes equation (flow across a cylinder).

The keyword arguments include:

`do_vis`  : Performs visualization of the desired quantity (such as fluid pressure, velocity magnitude) on a slice of the 3D staggered grid. By default, set to `false`. 
`do_save` : Saves three-dimensional arrays of the desired quantities at checkpoints.
"""
@views function navier_stokes_3D_solver_cylinder_xpu(; do_vis=false, do_save=true)
    # Physics
    ## dimensionally independent
    lz        = 1.0    # [m]
    ρ         = 1.0    # [kg/m^3]
    vin       = 1.0    # [m/s]
    ## scales
    psc       = ρ*vin^2
    ## nondimensional parameters
    Re        = 1e6    # rho*vsc*lz/μ
    Fr        = Inf    # vsc/sqrt(g*ly)
    lx_lz     = 0.5    # lx/lz
    ly_lz     = 0.5    # ly/lz
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
    μ         = 1/Re*ρ*vin*lz
    g         = 1/Fr^2*vin^2/lz
    r2        = (r_lz*lz)^2    
    # Numerics
    nz        = 255
    ny        = floor(Int,nz*ly_lz)
    nx        = floor(Int,nz*lx_lz)
    εit       = 1e-4
    niter     = 50*max(nx, ny, nz)
    nchk      = 1*(max(nx, ny, nz)-1)
    nvis      = 50
    nt        = 2000
    nsave     = 50
    CFLτ      = 0.9/sqrt(3) 
    CFL_visc  = 1/5.1
    CFL_adv   = 1.0 
    # Preprocessing
    dx,dy,dz  = lx/nx, ly/ny, lz/nz
    dt        = min(CFL_visc*dz^2*ρ/μ,CFL_adv*dz/vin)
    damp      = 2/nz
    dτ        = CFLτ*dz
    xc,yc,zc  = LinRange(-(lx-dx)/2,(lx-dx)/2,nx  ),LinRange(-(ly-dy)/2,(ly-dy)/2,ny  ), LinRange(-(lz-dz)/2,(lz-dz)/2,nz  )
    xv,yv,zv  = LinRange(-lx/2     ,lx/2     ,nx+1),LinRange(-ly/2     ,ly/2     ,ny+1), LinRange(-lz/2,     lz/2,     nz+1)
    # Array allocation
    Pr        = @zeros(nx  ,ny  ,nz  )
    dPrdτ     = @zeros(nx-2,ny-2,nz-2)
    τxx       = @zeros(nx  ,ny  ,nz  )
    τyy       = @zeros(nx  ,ny  ,nz  )
    τzz       = @zeros(nx  ,ny  ,nz  )
    τxy       = @zeros(nx-1,ny-1,nz-2)
    τxz       = @zeros(nx-1,ny-2,nz-1)
    τyz       = @zeros(nx-2,ny-1,nz-1)
    Vx        = @zeros(nx+1,ny  ,nz  )
    Vy        = @zeros(nx  ,ny+1,nz  )
    Vz        = @zeros(nx  ,ny  ,nz+1)
    Vx_o      = @zeros(nx+1,ny  ,nz  )
    Vy_o      = @zeros(nx  ,ny+1,nz  )
    Vz_o      = @zeros(nx  ,ny  ,nz+1)
    ∇V        = @zeros(nx  ,ny  ,nz  )
    Rp        = @zeros(nx-2,ny-2,nz-2)
    # Initialization
    x           = LinRange(0.5dx, lx-0.5dx,nx)
    y           = LinRange(0.5dy, ly-0.5dy,ny)
    Vprof       = Data.Array(@. 4*vin*x/lx*(1.0-x/lx) + 4*vin*y'/ly*(1.0-y'/ly))
    Vz[:,:,1]  .= Vprof
    Pr          = Data.Array([-(zc[iz] - lz/2)*ρ*g for ix=1:nx, iy=1:ny, iz=1:nz])
    # if do_save !ispath("./out_vis") && mkdir("./out_vis"); matwrite("out_vis/step_0.mat",Dict("Pr"=>Array(Pr),"Vx"=>Array(Vx),"Vy"=>Array(Vy), "Vz"=>Array(Vz),"C"=>Array(C),"dx"=>dx,"dy"=>dy, "dz"=>dz)) end
    # Iterative Solving
    for it = 1:nt
        err_evo = Float64[]; iter_evo = Float64[]
        # Velocity update, divergence update, sphere BC update
        @parallel update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)
        @parallel predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)
        @parallel set_cylinder!(Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2)
        @parallel update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)
        println("#it = $it")
        for iter = 1:niter
            # Pressure update by pseudo-transient solver
            @parallel update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)
            @parallel update_Pr!(Pr,dPrdτ,dτ)
            set_bc_Pr!(Pr, 0.0)
            if iter % nchk == 0
                # Error computation
                @parallel compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)
                err = maximum(abs.(Rp))*lz^2/psc
                push!(err_evo, err); push!(iter_evo,iter/nz)
                @printf("  #iter = %d, err = %1.3e\n", iter, err)
                if err < εit || !isfinite(err) break end
            end
        end
        # Velocity correction
        @parallel correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)
        @parallel set_cylinder!(Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2)
        set_bc_Vel!(Vx, Vy, Vz, Vprof)
        Vx_o .= Vx; Vy_o .= Vy; Vz_o .= Vz;
        # Advection scheme 
        @parallel advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,dt,dx,dy,dz)
        if do_vis && it % nvis == 0
            p1=heatmap(xc,yc,Array(Pr)';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Pr")
            p2=plot(iter_evo,err_evo;yscale=:log10)
            p3=heatmap(xc,yc,Array(C)';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="C")
            p4=heatmap(xc,yv,Array(Vy)';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vy")
            display(plot(p1,p2,p3,p4))
        end
        if do_save && it % nsave == 0
            save_array("./out_vis_cylinder/out_Vx_$it", convert.(Float32, Array(Vx)))
            save_array("./out_vis_cylinder/out_Vy_$it", convert.(Float32, Array(Vy)))
            save_array("./out_vis_cylinder/out_Vz_$it", convert.(Float32, Array(Vz)))
            save_array("./out_vis_cylinder/out_Pr_$it", convert.(Float32, Array(Pr)))
        end
    end
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
    @inn(Vx) = @inn(Vx) + dt/ρ*(@d_xi(τxx)/dx + @d_ya(τxy)/dy  + @d_za(τxz)/dz)
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
    bc_xyV!(A, V)

Sets zero-flux boundary condition in one xy-plane (index `end-1`) and a Dirichlet boundary condition = `V` in the 
other xy-plane (index `1`) for the quantity `A`. 
"""
@parallel_indices (ix, iy) function bc_xyV!(A, V)
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
    @parallel bc_yz!(Vy)
    @parallel bc_xy!(Vy)
    @parallel bc_xz!(Vz)
    @parallel bc_yz!(Vz)
    @parallel bc_xyV!(Vz, Vprof)
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

#Unit test.
"""
    backtrack!(A,A_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)

Performs backtracking for fluid velocities in the advection scheme.
"""
@inline function backtrack!(A,A_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    δx,δy,δz    = dt*vxc/dx, dt*vyc/dy, dt*vzc/dz
    ix1         = clamp(floor(Int,ix-δx),1,size(A,1))
    iy1         = clamp(floor(Int,iy-δy),1,size(A,2))
    iz1         = clamp(floor(Int,iz-δz),1,size(A,3))
    ix2,iy2,iz2 = clamp(ix1+1,1,size(A,1)),clamp(iy1+1,1,size(A,2)), clamp(iz1+1,1,size(A,3))
    δx = (δx>0) - (δx%1); δy = (δy>0) - (δy%1); δz = (δz>0) - (δz%1);
    fx11        = lerp(A_o[ix1,iy1,iz1], A_o[ix2,iy1,iz1],δx)
    fx21        = lerp(A_o[ix1,iy2,iz1], A_o[ix2,iy2,iz1],δx)
    fx12        = lerp(A_o[ix1,iy1,iz2], A_o[ix2,iy1,iz2],δx)
    fx22        = lerp(A_o[ix1,iy2,iz2], A_o[ix2,iy2,iz2],δx)
    fy1         = lerp(fx11, fx21, δy)
    fy2         = lerp(fx12, fx22, δy)
    A[ix,iy,iz] = lerp(fy1,fy2,δz)
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
@parallel_indices (ix,iy,iz) function advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,dt,dx,dy,dz)
    if ix > 1 && ix < size(Vx,1) && iy <= size(Vx,2) && iz <=size(Vx,3)
        vxc      = Vx_o[ix,iy,iz]
        vyc      = 0.25*(Vy_o[ix-1,iy,iz]+Vy_o[ix-1,iy+1,iz]+Vy_o[ix,iy,iz]+Vy_o[ix,iy+1,iz])
        vzc      = 0.25*(Vz_o[ix-1,iy,iz]+Vz_o[ix-1,iy,iz+1]+Vz_o[ix,iy,iz+1]+Vz_o[ix,iy,iz])
        backtrack!(Vx,Vx_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    if iy > 1 && iy < size(Vy,2) && ix <= size(Vy,1) && iz <= size(Vy,3)
        vxc      = 0.25*(Vx_o[ix,iy-1,iz]+Vx_o[ix+1,iy-1,iz]+Vx_o[ix,iy,iz]+Vx_o[ix+1,iy,iz])
        vyc      = Vy_o[ix,iy,iz]
        vzc      = 0.25*(Vz_o[ix,iy-1,iz]+Vz_o[ix,iy-1,iz+1]+Vz_o[ix,iy,iz]+Vz_o[ix,iy,iz+1])
        backtrack!(Vy,Vy_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    if iz > 1 && iz < size(Vz,3) && ix <= size(Vz,1) && iy <= size(Vz,2)
        vxc      = 0.25*(Vx_o[ix,iy,iz-1] + Vx_o[ix+1,iy,iz-1] + Vx_o[ix,iy,iz] + Vx_o[ix+1,iy,iz])
        vyc      = 0.25*(Vy_o[ix,iy,iz-1] + Vy_o[ix,iy+1,iz-1] + Vy_o[ix,iy,iz] + Vy_o[ix,iy+1,iz]) 
        vzc      = Vz_o[ix,iy,iz]
        backtrack!(Vz,Vz_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    return
end

"""
    set_cylinder!(Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2)

Sets a cylindrical object in the three-dimensional domain with a no-slip surface.
"""
@parallel_indices (ix,iy,iz) function set_cylinder!(Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2)
    xv,yv,zv = (ix-1)*dx - lx/2, (iy-1)*dy - ly/2, (iz-1)*dz -lz/2
    xc,yc,zc = xv+dx/2, yv+dx/2 , zv+dz/2 
    if checkbounds(Bool,Vx,ix,iy,iz)
        xr = (xc-ox)
        yr = (yc-oy)
        zr = (zc-oz)
        if xr*xr/r2 + zr*zr/r2 < 1.0 && yr >= -0.1 && yr <= 0.1
            Vx[ix,iy,iz] = 0.0
        end
    end
    if checkbounds(Bool,Vy,ix,iy,iz)
        xr = (xc-ox)
        yr = (yc-oy)
        zr = (zc-oz)
        if xr*xr/r2 + zr*zr/r2 < 1.0 && yr >= -0.1 && yr <= 0.1
            Vy[ix,iy,iz] = 0.0
        end
    end
    if checkbounds(Bool,Vz,ix,iy,iz)
        xr = (xc-ox)
        yr = (yc-oy)
        zr = (zc-oz)
        if xr*xr/r2 + zr*zr/r2 < 1.0 && yr >= -0.1 && yr <= 0.1
            Vz[ix,iy,iz] = 0.0
        end
    end
    return
end

# Solver Call
navier_stokes_3D_solver_cylinder_xpu()