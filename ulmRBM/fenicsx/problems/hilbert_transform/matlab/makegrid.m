function grid = makegrid(t)
% returns a struct with grid parameters from time steps t
% Input: 
% t ... vector with time nodes
% Output: 
% grid ... struct with values
% .t ... time step nodes
% .N ... number of elements
% .tmid ... midpoints of elements
% .h ... times step sizes 
% .hmax , .hmin ... maximum/minimum size of time steps
% .e ... edges with vertex numbers


grid = struct; 
grid.t = t; 
grid.N = length(t)-1; %number of elements
grid.tmid=(t(1:end-1)+t(2:end))/2; % midpoints of elements
grid.h = t(2:end)-t(1:end-1); % time step sizes 
grid.hmax = max(grid.h); grid.hmin = min(grid.h);         
grid.e = [(1:grid.N);(2:grid.N+1)]; % edge numbers for time (t is ordered)


end