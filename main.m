A=imread('F:\matlab\NMSI\NMI\1\052_01_01_051_08.png');
B=imread('F:\matlab\NMSI\NMI\2\052_01_01_051_08.png');
tic 
[mi1,nmi1]=NormMutualInfo(A,B,2)
toc
