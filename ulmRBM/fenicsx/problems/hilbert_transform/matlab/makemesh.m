function mesh = makemesh(x)
% returns a struct with grid parameters from time steps t
% Input: 
% x ... vector with coordinates of vertices
% Output: 
% mesh ... struct with values
% .p ... x
% .N ... number of elements
% .np ... number of Dofs
% .h ... times step sizes 
% .hmax , .hmin ... maximum/minimum size of time steps
% .t ... edges with vertex numbers
% .e ... boundary edges


mesh = struct; 
mesh.p = x; 
mesh.N = length(x)-1; %number of elements
mesh.h = x(2:end)-x(1:end-1); % time step sizes 
mesh.hmax = max(mesh.h); mesh.hmin = min(mesh.h);         
mesh.t = [(1:mesh.N);(2:mesh.N+1)]; % edge numbers for time (t is ordered)
mesh.np = mesh.N+1; 
mesh.e = [1,mesh.np]; 
end