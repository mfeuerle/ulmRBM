function L_ht = hilbert_Lmatrix(t,A_ht00)
% returns the matrix for the realisation of the inner product of
% <H_T dtu, dtv > for u piecewise linear and v piecewise linear
% Input: 
% t ... vector with time steps
% A_htxx ... auxiliary matrices
% Author: Richard Löscher, 17.11.2022

% auxiliary matrices 
Z_ht1 = zfullmatrix(t,1);

L_ht = (Z_ht1*A_ht00*Z_ht1')';

end