using Test
using MPI

MPI.Init()

"""
This code enables testing of the Navier Stokes solver in 3D. The following features are tested: 

- saving and reading of arrays (to ensure that the arrays are correctly saved even in a multi xpu context, as even the testing can be run on multixpu)
- setting the sphere in a three dimensional context. This is asserted by means of initializing the concentration to 1 in the entire domain and setting the sphere, located at 
  the origin of the domain to zero. This is even more important in a multixpu context, and therefore the tests can be run on multixpu (simply setting USE_GPU to true inside 
  'NavierStokes_arthur_multi_xpu_testing.jl'). 
- a reference test of a simple context with nz=63, nx=ny=31 grid points, for nt=10 time steps. The pressure and velocity fields are compared to the ground truth obtained
  in previous implementations (in order to check if modifications lead or not to unexpected changes in these quantities)
"""



"""
Reference test: Testing pressure and velocities with respect to ground truth
"""


println("Starting reference test")
include("./NavierStokes_arthur_multi_xpu_testing.jl")
println("Reference test passed")


"""
Unit test: Testing saving and loading of arrays, asserting that the arrays are correctly read, especially on multpixpu
"""
nx = 8
ny = 8
nz = 16
me, dims,nprocs,coords  = init_global_grid(nx, ny, nz;init_MPI=false)
coords    = Data.Array(coords)
if me==0
    array = [i + j + k for i in 1:nx, j in 1:ny, k in 1:nz]
    A_loc=Data.Array(array)
else
    A_loc=@zeros(nx,ny,nz)
end
A_loc=A_loc.*2
update_halo!(A_loc)
nx_v, ny_v, nz_v = (nx - 2) * dims[1], (ny - 2) * dims[2], (nz - 2) * dims[3]
A_inn = zeros(nx - 2, ny - 2, nz - 2) 
A_save  = zeros(nx_v, ny_v, nz_v)

A_inn.= Array(A_loc[2:end-1, 2:end-1, 2:end-1]); gather!(A_inn, A_save)
if me==0
    save_array("./out/A_out", A_save)
end
B=zeros(nx_v,ny_v,nz_v)
B_exact= [i + j + k for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1]
B_exact.=B_exact.*2

if me==0
    load_array("./out/A_out",B)
    println("Testing saving and loading of arrays")
    @test B[1:nx-2,1:ny-2,1:nz-2]==B_exact
    if dims[1]>1 # if testing is performed on more than one GPU
        @test all(B[nx-1:end,ny-1:end,nz-1:end].==0)
    end
    println("Test passed")
end


"""
Unit test: Testing setting sphere method, especially useful on multixpu, asserting if the sphere is indeed centered at the origin of the domain. 
"""
Vx        = @zeros(nx+1,ny,nz)
Vy        = @zeros(nx  ,ny+1,nz)
Vz        = @zeros(nx  ,ny, nz+1)
C         = ones(Float64, nx,ny,nz)
C         = Data.Array(C)

lz        = 1.0 
lx_lz     = 0.5 
ly_lz     = 0.5
r_lz      = 0.05
ox_lz     = 0.0
oy_lz     = 0.0
oz_lz     = 0.0
lx        = lx_lz*lz
ly        = ly_lz*lz
ox        = ox_lz*lz
oy        = oy_lz*lz
oz        = oz_lz*lz
r2        = (r_lz*lz)^2
dx,dy,dz  = lx/nx_g(),ly/ny_g(),lz/nz_g()
@parallel set_sphere!(C,Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2,nx,ny,nz,coords)

C_inn=zeros(nx-2,ny-2,nz-2)
C_glob=zeros(nx_v, ny_v, nz_v)
C_inn.= Array(C[2:end-1, 2:end-1, 2:end-1]); gather!(C_inn, C_glob)

if me==0
    println("Testing setting sphere inside the domain")
    @test C_glob[Int(nx_v/2),Int(ny_v/2),Int(nz_v/2)]==0.0
    println("Test passed")
end

MPI.Finalize()




