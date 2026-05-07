function S = skew_n(x)
% SKEW_N Gera uma matriz antissimétrica nxn (elemento de so(n))
%
%   S = skew_n(omega)
%
%   Entrada:
%       omega - vetor coluna de tamanho n*(n-1)/2
%
%   Saída:
%       S - matriz antissimétrica nxn (S = -S')
%
%   A dimensão n é inferida automaticamente de omega.

    import casadi.*
%    v1 = x(1);
%    v2 = x(2);
%    omega = x(3);
%    a1 = x(4);
%    a2 = x(5);
%    alpha = x(6);
% 
%    S = [0,-omega,v1,a1;omega,0,v2,a2;0,0,0,alpha;0,0,0,0];