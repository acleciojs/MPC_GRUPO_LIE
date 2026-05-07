function [J] = Jacobian_f(f, R_list, S_sym, omega_sym)
% SYMBOLIC_JACOBIAN_F_MULTI_CASADI
% Jacobiana simbólica em SO(n) usando S(omega) simbólico
%
% f         : vetor SX/MX (M×1)
% R_list    : cell array com matrizes simbólicas n×n
% S_sym     : matriz simbólica n×n dependente de omega
% omega_sym : variável simbólica (m×1)
%
% Saídas:
% J      : Jacobiana simbólica
% J_fun  : função CasADi da Jacobiana
% S_fun  : função CasADi do mapeamento S(omega)

import casadi.*

%% =========================================================
%% Dimensões
%% =========================================================
N = numel(R_list);
M = numel(f);
m = length(omega_sym);
d = m;

%% =========================================================
%% Criar função S(omega)
%% =========================================================
S_fun = Function('S_fun', {omega_sym}, {S_sym});

%% =========================================================
%% Variável escalar
%% =========================================================
e = SX.sym('e');

%% =========================================================
%% Base canônica
%% =========================================================
I = SX.eye(m);

%% =========================================================
%% Jacobiana total
%% =========================================================
JR = SX.zeros(M, d*N);

%% =========================================================
%% Loop principal
%% =========================================================
for r = 1:N

    R = R_list{r};
    J_r = SX.zeros(M, d);

    for k = 1:d

        % direção canônica
        omega_k = I(:,k);

        % ===== usa função criada automaticamente =====
        S = S_fun(omega_k);

        % perturbação exponencial
        R_e = R * expm_casadi(S * e);

        % substituição
        f_e = substitute(f, R, R_e);

        % derivada direcional
        dFde = jacobian(f_e, e);

        % avaliar em e = 0
        J_r(:,k) = substitute(dFde, e, 0);
    end

    % armazenar bloco
    JR(:, (r-1)*d + 1 : r*d) = J_r;
end

J = JR;

%% =========================================================
%% Criar função final da Jacobiana
%% =========================================================
inputs = {};
for r = 1:N
    inputs{end+1} = R_list{r};
end


end