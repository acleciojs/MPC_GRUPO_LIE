function L = logm_casadi(X, order, s)
% LOGM_CASADI ESTÁVEL
% Compatível com CasADi (SX/MX)
%
% Usa:
% 1) scaling: X -> X^(1/2^s)
% 2) Mercator na matriz próxima da identidade
% 3) desescala multiplicando por 2^s

    if nargin < 2
        order = 20;
    end
    if nargin < 3
        s = 3;   % número de escalonamentos
    end

    n = size(X,1);
    I = eye(n);

    % === STEP 1: calcular raiz 2^s vezes ===
    Y = X;
    for i = 1:s
        Y = sqrtm_newton(Y,5);   % 5 iterações Newton
    end

    % === STEP 2: Mercator expansion ===
    Z = Y - I;

    L = 0*X;
    Zk = Z;

    for k = 1:order
        L = L + ((-1)^(k+1)/k)*Zk;
        Zk = Zk*Z;
    end

    % === STEP 3: desfaz scaling ===
    L = (2^s)*L;
end
function Y = sqrtm_newton(X, iters)
% Raiz quadrada matricial via Newton

    n = size(X,1);
    Y = X; 
    I = eye(n);

    for k = 1:iters
        Y = 0.5*(Y + X*inv(Y));
    end
end
