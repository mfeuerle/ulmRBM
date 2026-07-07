function [M_ht] = assemble_massmatrix_hilbert(grid)
% Builds the L2 mass matrix of 1D mesh with the Hilbert transform
%   M(i,j) = (H_T phi_j,  phi_i)_L2
% for continous picewise linear basis functions phi (aka hat functions).

rho = 10;

A_ht00=arqmatrix(grid,0,0,rho);
A_ht01=arqmatrix(grid,0,1,rho);
A_ht10=arqmatrix(grid,1,0,rho);
A_ht11=arqmatrix(grid,1,1,rho);
M_ht = hilbert_Mmatrix(grid.t,A_ht00,A_ht10,A_ht01,A_ht11);

end