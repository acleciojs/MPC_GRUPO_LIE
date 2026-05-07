function S = skew_n2(x)
% SKEW_N Gera uma matriz antissimétrica nxn (elemento de so(n))
%
%   S = skew_n(x)
%
%   Entrada:
%       x - vetor coluna de tamanho n*(n-1)/2
%
%   Saída:
%       S - matriz antissimétrica nxn (S = -S')

    import casadi.*

    % Tamanho do vetor
    m = length(x);

    % Resolver n a partir de m = n*(n-1)/2
    n = (1 + sqrt(1 + 8*m)) / 2;

    if abs(n - round(n)) > 1e-10
        error('Dimensão inválida: tamanho do vetor não corresponde a n(n-1)/2');
    end
    n = round(n);

    % Inicializa matriz
    S = SX.zeros(n,n);

    % Preenchimento
    k = 1;
    for i = 1:n
        for j = i+1:n
            S(i,j) = x(k);
            S(j,i) = -x(k);
            k = k + 1;
        end
    end
end