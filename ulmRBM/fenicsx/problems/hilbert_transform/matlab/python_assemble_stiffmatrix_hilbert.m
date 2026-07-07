function [] = python_assemble_stiffmatrix_hilbert(t)
A  = assemble_stiffmatrix_hilbert(makegrid(t));
[i,j,data] = find(A);
all = [i,j,data];
save('.stima.txt','all','-ascii','-double')
return