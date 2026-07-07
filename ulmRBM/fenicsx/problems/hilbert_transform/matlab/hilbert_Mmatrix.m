function M_ht = hilbert_Mmatrix(t,A_ht00,A_ht10,A_ht01,A_ht11)
% returns the matrix for the realisation of the inner product of
% <u,H_T v > for u piecewise linear and v piecewise linear
% Input: 
% t ... vector with time steps
% A_htxx ... auxiliary matrices
% Author: Richard Löscher, 17.11.2022

% auxiliary matrices
Z_ht0 = zfullmatrix(t,0); 
Z_ht1 = zfullmatrix(t,1);

M_ht = (Z_ht1*A_ht11+Z_ht0*A_ht01)*Z_ht1'+(Z_ht1*A_ht10+Z_ht0*A_ht00)*Z_ht0';

end