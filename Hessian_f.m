function H = Hessian_f(f, G_list, S_sym, omega_sym)
% Hessiana simbólica (CasADi) compatível com R_i ∈ SO(n)
% Baseada em variações exponenciais e skew_n
%
% f      : função escalar SX/MX
% R_list : cell array com matrizes SX/MX n×n

import casadi.*

%% --- Verificação ---
if numel(f) ~= 1
    error('f deve ser escalar para calcular a Hessiana.');
end

%% --- Jacobiana total (1 × dN) ---
J = Jacobian_f(f, G_list,S_sym, omega_sym);

%% --- Dimensões ---
N = numel(G_list);          % número de rotações
n = size(G_list{1}, 1);     % dimensão de cada R
m = length(omega_sym);
d = m;                      % mesma escolha da Jacobiana
total_vars = d * N;

%% --- Inicializa Hessiana ---
% Preserva o tipo (SX ou MX)
if isa(f, 'casadi.SX')
    H = SX.zeros(total_vars, total_vars);
else
    H = MX.zeros(total_vars, total_vars);
end

%% --- Cada linha da Hessiana ---
for i = 1:total_vars
    % J(i) é escalar → Jacobiana retorna 1 × (dN)
    Hi = Jacobian_f(J(i), G_list,S_sym, omega_sym);
    H(i, :) = Hi;
end

end
