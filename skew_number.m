function S = skew_number(omega)
% SKEW_NUMBER Gera matriz antissimétrica nxn (elemento de so(n))
%
%   S = skew_number(omega)
%
%   Entrada:
%       omega - vetor coluna de tamanho n*(n-1)/2
%
%   Saída:
%       S - matriz antissimétrica nxn (S = -S')

    m = length(omega);

    % Resolve n(n-1)/2 = m
    n = (1 + sqrt(1 + 8*m)) / 2;

    if abs(n - round(n)) > 1e-12
        error('O tamanho de omega deve ser n*(n-1)/2.');
    end

    n = round(n);

    % Inicializa matriz nxn
    S = zeros(n);

    % Preenche parte superior
    k = 1;
    for i = 1:n
        for j = i+1:n
            S(i,j) = -omega(k);
            S(j,i) =  omega(k);
            k = k + 1;
        end
    end
end