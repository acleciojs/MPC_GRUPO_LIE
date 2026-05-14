function S = skew_n(omega)
% SKEW_N  Mapa so(n): vetor omega -> matriz antissimétrica S(omega)
%
% Entrada:
%   omega : CasADi SX ou MX, tamanho n(n-1)/2
%
% Saída:
%   S : matriz nxn em so(n)

    import casadi.*

    omega = omega(:); % garante vetor coluna simbólico

    m = length(omega);

    % resolve n(n-1)/2 = m
    n = (1 + sqrt(1 + 8*m)) / 2;

    % checagem simbólica (vai funcionar em build-time)
    if abs(double(n) - round(double(n))) > 1e-10
        error('omega não tem dimensão compatível com so(n)');
    end

    n = round(n);

    % matriz simbólica CasADi
    S = SX.zeros(n,n);   % (pode trocar SX por MX se quiser)

    k = 1;
    for i = 1:n
        for j = i+1:n
            S(i,j) = omega(k);
            S(j,i) = -omega(k);
            k = k + 1;
        end
    end
end