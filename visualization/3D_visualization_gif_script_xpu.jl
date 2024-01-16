using GLMakie, Printf 

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

function visualise_velocity_mag(iframe = 0, vis_save = 0)
    lx, ly, lz = 0.5, 0.5, 1.0
    nz      = 255
    nx      = ny = 127
    Vx      = zeros(Float32, nx+1, ny, nz)
    Vy      = zeros(Float32, nx, ny+1, nz)
    Vz      = zeros(Float32, nx, ny, nz+1)
    Vx_c    = zeros(Float32, nx, ny, nz  )
    Vy_c    = zeros(Float32, nx, ny, nz  )
    Vz_c    = zeros(Float32, nx, ny, nz  )
    Vmag    = zeros(Float32, nx, ny, nz  )
    load_array("../scripts/out_vis_all_xpu/out_Vx_$(iframe)", Vx)
    load_array("../scripts/out_vis_all_xpu/out_Vy_$(iframe)", Vy)
    load_array("../scripts/out_vis_all_xpu/out_Vz_$(iframe)", Vz)
    Vx_c .= avx(Array(Vx))
    Vy_c .= avy(Array(Vy))
    Vz_c .= avz(Array(Vz))
    Vmag .= sqrt.(Vx_c .^ 2 .+ Vy_c .^ 2 .+ Vz_c .^ 2)
    xc, yc, zc = LinRange(-lx/2 ,lx/2 ,nx+1),LinRange(-ly/2,ly/2,ny+1), LinRange(-lz/2, lz/2, nz+1)
    fig = Figure(resolution=(1600, 1000), fontsize=24)
    ax  = Axis3(fig[1, 1]; aspect=(0.5, 0.5, 1.0), title="Velocity Magnitude", xlabel="lx", ylabel="ly", zlabel="lz")
    surf_T = contour!(ax, xc, yc, zc, Vmag; alpha=0.05, colormap=:summer)
    if !isdir("./3D_gif_velocity_mag_s_xpu")
        mkdir("./3D_gif_velocity_mag_s_xpu")
    end
    save(@sprintf("./3D_gif_velocity_mag_s_xpu/%06d.png", vis_save), fig)
    return fig
end

function visualise_pressure(iframe = 0, vis_save = 0)
    lx, ly, lz = 0.5, 0.5, 1.0
    nz = 255
    nx = ny = 127
    Pr  = zeros(Float32, nx, ny, nz)
    load_array("../scripts/out_vis_all_xpu/out_Pr_$(iframe)", Pr)
    xc, yc, zc = LinRange(-lx/2 ,lx/2 ,nx+1),LinRange(-ly/2,ly/2,ny+1), LinRange(-lz/2, lz/2, nz+1)
    fig = Figure(resolution=(1600, 1000), fontsize=24)
    ax  = Axis3(fig[1, 1]; aspect=(0.5, 0.5, 1.0), title="Pressure Field", xlabel="lx", ylabel="ly", zlabel="lz")
    surf_T = contour!(ax, xc, yc, zc, Pr; alpha=0.05, colormap=:turbo)
    if !isdir("./3D_gif_pressure_s_xpu")
        mkdir("./3D_gif_pressure_s_xpu")
    end
    save(@sprintf("./3D_gif_pressure_s_xpu/%06d.png", vis_save), fig)
    return fig
end

function visualise_vorticity(iframe = 0, vis_save = 0)
    lx, ly, lz = 0.5, 0.5, 1.0
    nz = 255
    nx = ny = 127
    dx,dy,dz  = lx/nx, ly/ny, lz/nz
    ω   = zeros(Float32, nx-2, ny-2, nz-2)
    ωx  = zeros(Float32, nx-2, ny-2, nz-2)
    ωy  = zeros(Float32, nx-2, ny-2, nz-2)
    ωz  = zeros(Float32, nx-2, ny-2, nz-2)
    Vx  = zeros(Float32, nx+1, ny, nz)
    Vy  = zeros(Float32, nx, ny+1, nz)
    Vz  = zeros(Float32, nx, ny, nz+1)
    load_array("../scripts/out_vis_all_xpu/out_Vx_$(iframe)", Vx)
    load_array("../scripts/out_vis_all_xpu/out_Vy_$(iframe)", Vy)
    load_array("../scripts/out_vis_all_xpu/out_Vz_$(iframe)", Vz)
    ωx .= avz(avy(diff(Vz, dims = 2)))[2:end-1,:,2:end-1]./dy .- avz(avy(diff(Vy, dims = 3)))[2:end-1,2:end-1,:]./dz
    ωy .= avz(avx(diff(Vx, dims = 3)))[2:end-1,2:end-1,:]./dz .- avz(avx(diff(Vz, dims = 1)))[:,2:end-1,2:end-1]./dx
    ωz .= avx(avy(diff(Vy, dims = 1)))[:,2:end-1,2:end-1]./dx .- avx(avy(diff(Vx, dims = 2)))[2:end-1,:,2:end-1]./dy
    ω  .= sqrt.(ωx .^ 2 .+ ωy .^ 2 .+ ωz .^ 2)
    xc, yc, zc = LinRange(-lx/2 ,lx/2 ,nx+1),LinRange(-ly/2,ly/2,ny+1), LinRange(-lz/2, lz/2, nz+1)
    fig = Figure(resolution=(1600, 1000), fontsize=24)
    ax  = Axis3(fig[1, 1]; aspect=(0.5, 0.5, 1.0), title="Vorticity", xlabel="lx", ylabel="ly", zlabel="lz")
    surf_T = contour!(ax, xc, yc, zc, ω; alpha=0.05, colormap=:cool)
    if !isdir("./3D_gif_vorticity_s_xpu")
        mkdir("./3D_gif_vorticity_s_xpu")
    end
    save(@sprintf("./3D_gif_vorticity_s_xpu/%06d.png", vis_save), fig)
    return fig
end

nvis    = 1
nt      = 20
frames  = 50
for it = 1:nt
    visualise_velocity_mag(it*frames, it)
    visualise_pressure(it*frames, it)
    visualise_vorticity(it*frames, it)
end 

import Plots:Animation, buildanimation                  
fnames = [@sprintf("%06d.png", k) for k in 1:nt]

anim_Vmag = Animation("./3D_gif_velocity_mag_s_xpu", fnames); 
buildanimation(anim_Vmag, "./3D_gif_velocity_mag_s_xpu/3D_Vmag.gif", fps = 5, show_msg=false)    

anim_p = Animation("./3D_gif_pressure_s_xpu", fnames); 
buildanimation(anim_p, "./3D_gif_pressure_s_xpu/3D_pressure.gif", fps = 5, show_msg=false)

anim_ω = Animation("./3D_gif_vorticity_s_xpu", fnames); 
buildanimation(anim_ω, "./3D_gif_vorticity_s_xpu/3D_vorticity.gif", fps = 5, show_msg=false)  