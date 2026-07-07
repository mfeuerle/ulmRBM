function [] = python_assemble_massmatrix_hilbert(t)
A  = assemble_massmatrix_hilbert(makegrid(t));
[i,j,data] = find(A);
all = [i,j,data];
save('.masma.txt','all','-ascii','-double')
return