function [L_ht] = assemble_stiffmatrix_hilbert(grid)
% Builds the L2 mass matrix of 1D mesh with the Hilbert transform
%   L(i,j) = (H_T grad(phi_j), grad(phi_i))_L2
% for continous picewise linear basis functions phi (aka hat functions).

rho = 10;

A_ht00=arqmatrix(grid,0,0,rho);
L_ht = hilbert_Lmatrix(grid.t,A_ht00);

end