function J = symbolic_Jacobian_f_multi_casadi(f, R_list)
% Jacobiana simbólica (CasADi) em relação a R ∈ SO(n)
% Usa variação R * expm_so_n_casadi(S*e)
%
% f      : vetor SX/MX M×1
% R_list : cell array com matrizes SX/MX n×n

import casadi.*

%% Dimensões
N = numel(R_list);          % número de matrizes R
M = numel(f);               % dimensão da função
n = size(R_list{1},1);      % dimensão de SO(n)
m = (n*(n-1))/2;
% Número de direções na álgebra (usa base canônica)
d = m;

%% Variável escalar de perturbação
e = SX.sym('e');

%% Jacobiana total
JR = SX.zeros(M, d*N);

%% Identidade

I = SX.eye(m);

%% Loop sobre cada matriz de rotação
for r = 1:N

    R = R_list{r};
    J_r = SX.zeros(M, d);

    %% Loop sobre direções canônicas da álgebra
    for k = 1:d

        % Direção canônica
        omega = I(:,k);

        % Gerador antissimétrico
        S = skew_n(omega);   % deve retornar SX n×n

        % Variação exponencial (CasADi)
        R_e = R * expm_casadi(S * e);

        % Substitui SOMENTE esta matriz em f
       
        f_e = substitute(f, R, R_e);

        % Derivada direcional via Jacobiana em e
        dFde = jacobian(f_e, e);

        % Avaliação em e = 0
        J_r(:,k) = substitute(dFde, e, 0);
    end

    % Armazena bloco
    JR(:, (r-1)*d + 1 : r*d) = J_r;
end


%% Jacobiana final
J = JR;

end
