using Plots, Printf

"""
    av1(A)

Returns an array computed by averaging the array `A` in all dimensions (applicable to n-dimension arrays).

#Example
```jldoctest
julia> av1([10,20,30])
2-element Vector{Float64}:
 15.0
 25.0
```
"""
@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])

"""
    avx(A)

Returns an array computed by averaging the array `A` in the first dimension (applicable to 3-dimension arrays).

#Example
```jldoctest
julia> avx([[10 20; 30 40];;;[50 60; 70 80]])
1×2×2 Array{Float64, 3}:
[:, :, 1] =
 20.0  30.0

[:, :, 2] =
 60.0  70.0
```
"""
@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])

"""
    avy(A)

Returns an array computed by averaging the array `A` in the second dimension (applicable to 3-dimension arrays).

#Example
```jldoctest
julia> avy([[10 20; 30 40];;;[50 60; 70 80]])
2×1×2 Array{Float64, 3}:
[:, :, 1] =
 15.0
 35.0

[:, :, 2] =
 55.0
 75.0
```
"""
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])

"""
    avz(A)

Returns an array computed by averaging the array `A` in the third dimension (applicable to 3-dimension arrays).

#Example
```jldoctest
julia> avz([[10 20; 30 40];;;[50 60; 70 80]])
2×2×1 Array{Float64, 3}:
[:, :, 1] =
 30.0  40.0
 50.0  60.0
```
"""
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])

function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid=open(fname, "r"); read!(fid, A); close(fid)
end

function visualise_velocity_mag_slice(iframe = 0, vis_save = 0)
    lx, ly, lz = 0.5, 0.5, 1.0
    nz      = 506
    nx      = ny = 250
    Vx      = zeros(Float32, nx, ny, nz)
    Vy      = zeros(Float32, nx, ny, nz)
    Vz      = zeros(Float32, nx, ny, nz)
    Vmag    = zeros(Float32, nx, ny, nz)
    load_array("../scripts/out_vis_all/out_Vx_$(iframe)", Vx)
    load_array("../scripts/out_vis_all/out_Vy_$(iframe)", Vy)
    load_array("../scripts/out_vis_all/out_Vz_$(iframe)", Vz)
    Vx     .= Array(Vx)
    Vy     .= Array(Vy)
    Vz     .= Array(Vz)
    Vmag .= sqrt.(Vx .^ 2 .+ Vy .^ 2 .+ Vz .^ 2)
    xc, yc, zc = LinRange(-lx/2 ,lx/2 ,nx+1),LinRange(-ly/2,ly/2,ny+1), LinRange(-lz/2, lz/2, nz+1)
    p1=heatmap(xc,zc,Vmag[:, ceil(Int, ny/ 2), :]';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),title="Velocity Magnitude", c=:summer, clims=(0,2.5))
    if !isdir("./slice_velocity_mag_s_multixpu")
        mkdir("./slice_velocity_mag_s_multixpu")
    end
    png(p1, @sprintf("./slice_velocity_mag_s_multixpu/%06d.png", vis_save))
    return
end

function visualise_pressure_slice(iframe = 0, vis_save = 0)
    lx, ly, lz = 0.5, 0.5, 1.0
    nz = 506
    nx = ny = 250
    Pr  = zeros(Float32, nx, ny, nz)
    load_array("../scripts/out_vis_all/out_Pr_$(iframe)", Pr)
    xc, yc, zc = LinRange(-lx/2 ,lx/2 ,nx+1),LinRange(-ly/2,ly/2,ny+1), LinRange(-lz/2, lz/2, nz+1)
    p1=heatmap(xc,zc,Pr[:, ceil(Int, ny/ 2), :]';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),title="Pressure Field", c=:turbo, clims=(-2,2))
    if !isdir("./slice_pressure_s_multixpu")
        mkdir("./slice_pressure_s_multixpu")
    end
    png(p1, @sprintf("./slice_pressure_s_multixpu/%06d.png", vis_save))
    return
end

function visualise_vorticity_slice(iframe = 0, vis_save = 0)
    lx, ly, lz = 0.5, 0.5, 1.0
    nz = 506
    nx = ny = 250
    dx,dy,dz  = lx/nx, ly/ny, lz/nz
    ω   = zeros(Float32, nx-2, ny-2, nz-2)
    ωx  = zeros(Float32, nx-2, ny-2, nz-2)
    ωy  = zeros(Float32, nx-2, ny-2, nz-2)
    ωz  = zeros(Float32, nx-2, ny-2, nz-2)
    Vx  = zeros(Float32, nx, ny, nz)
    Vy  = zeros(Float32, nx, ny, nz)
    Vz  = zeros(Float32, nx, ny, nz)
    load_array("../scripts/out_vis_all/out_Vx_$(iframe)", Vx)
    load_array("../scripts/out_vis_all/out_Vy_$(iframe)", Vy)
    load_array("../scripts/out_vis_all/out_Vz_$(iframe)", Vz)
    ωx .= avy(diff(Vz, dims = 2))[2:end-1,:,2:end-1]./dy .- avz(diff(Vy, dims = 3))[2:end-1,2:end-1,:]./dz
    ωy .= avz(diff(Vx, dims = 3))[2:end-1,2:end-1,:]./dz .- avx(diff(Vz, dims = 1))[:,2:end-1,2:end-1]./dx
    ωz .= avx(diff(Vy, dims = 1))[:,2:end-1,2:end-1]./dx .- avy(diff(Vx, dims = 2))[2:end-1,:,2:end-1]./dy
    ω  .= sqrt.(ωx .^ 2 .+ ωy .^ 2 .+ ωz .^ 2)
    xc, yc, zc = LinRange(-lx/2 + dx ,lx/2 - dx ,nx-1),LinRange(-ly/2 + dy, ly/2 - dy, ny-1), LinRange(-lz/2 + dz, lz/2 - dz, nz-1)
    p1=heatmap(xc,zc,ω[:, ceil(Int, ny/ 2), :]';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),title="Vorticity",c=:cool, clims=(0,600))
    if !isdir("./slice_vorticity_s_multixpu")
        mkdir("./slice_vorticity_s_multixpu")
    end
    png(p1, @sprintf("./slice_vorticity_s_multixpu/%06d.png", vis_save))
    return
end


nvis    = 1
nt      = 20
frames  = 100
for it = 1:nt
    visualise_velocity_mag_slice(it*frames, it)
    visualise_pressure_slice(it*frames, it)
    visualise_vorticity_slice(it*frames, it)
end 

import Plots:Animation, buildanimation 
fnames = [@sprintf("%06d.png", k) for k in 1:nt]

anim_Vmag = Animation("./slice_velocity_mag_s_multixpu/", fnames); 
buildanimation(anim_Vmag, "./slice_velocity_mag_s_multixpu/navier_stokes_slice_Vmag_multigpu.gif", fps = 5, show_msg=false)  

anim_p = Animation("./slice_pressure_s_multixpu/", fnames); 
buildanimation(anim_p, "./slice_pressure_s_multixpu/navier_stokes_slice_pressure_multigpu.gif", fps = 5, show_msg=false)

anim_ω = Animation("./slice_vorticity_s_multixpu/", fnames); 
buildanimation(anim_ω, "./slice_vorticity_s_multixpu/navier_stokes_slice_vorticity_multigpu.gif", fps = 5, show_msg=false)