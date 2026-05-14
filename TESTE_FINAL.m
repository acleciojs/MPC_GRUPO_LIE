clc; clear all;
import casadi.*

%% =========================================================
% DADOS DO PROBLEMA DE WAHBA EM 7 DIMENSÕES
%% =========================================================
n_dim = 7;
n_pts = 10;

S_true = randn(n_dim, n_dim);
S_true = (S_true - S_true')/2;
M_true = expm(S_true);

V = randn(n_dim, n_pts);
V = V ./ vecnorm(V);

V_star = M_true * V;

%% =========================================================
% VARIÁVEIS DE DECISÃO
%% =========================================================
d = n_dim * (n_dim - 1) / 2;

x_sym = SX.sym('x', d, 1);

S_sym = skew_n(x_sym);

M_sym = SX.sym('M', n_dim, n_dim);

M_cell = {M_sym};

T0 = SX.sym('T0', n_dim^2, 1);

%% =========================================================
% FUNÇÃO OBJETIVO
%% =========================================================
custo = 0;

for j = 1:n_pts

    erro = V_star(:, j) - M_sym * V(:, j);

    custo = custo + dot(erro, erro);

end

%% =========================================================
% RESTRIÇÕES
%% =========================================================

% g(x) <= 0

g_ineq1 = -M_sym(n_dim, n_dim);

g_ineq2 = 1 - trace(M_sym);

u = zeros(n_dim,1);
u(1) = 1;

u0 = u;

g_ineq3 = cosd(60) - (M_sym*u)'*u0;

g = [g_ineq1;
     g_ineq2;
     g_ineq3];

h = [];

%% =========================================================
% SOLVER
%% =========================================================
[sym_data, casadi_funs] = solver_IPM( ...
    custo, M_cell, T0, S_sym, x_sym, g, h);

%% =========================================================
% CHUTE INICIAL
%% =========================================================

x0 = zeros(d,1);

%% =========================================================
% GERA MATRIZ INICIAL EM SO(n)
%% =========================================================

S0 = skew_number(0.1*randn(d,1));

M0 = expm(S0);

X0 = {M0};

T_real = [];

%% =========================================================
% VERIFICAÇÃO DO X0
%% =========================================================

fprintf('\n========================================\n');
fprintf('VERIFICAÇÃO DO CHUTE INICIAL\n');
fprintf('========================================\n');

ortho_err = norm(M0'*M0 - eye(n_dim), 'fro');

fprintf('Erro ortogonalidade: %.3e\n', ortho_err);

detM = det(M0);

fprintf('det(M0): %.6f\n', detM);

g1 = -M0(n_dim,n_dim);

g2 = 1 - trace(M0);

g3 = cosd(60) - (M0*u)'*u;

g_all = [g1; g2; g3];

fprintf('\nRestrições g(x)<=0:\n');

for i = 1:length(g_all)

    fprintf('g(%d) = %+ .6e\n', i, g_all(i));

end

if all(g_all <= 0)

    fprintf('\nX0 É VIÁVEL.\n');

else

    fprintf('\nX0 NÃO É VIÁVEL.\n');

end

%% =========================================================
% OTIMIZAÇÃO
%% =========================================================

[X_opt_cell] = run_optimization( ...
    sym_data, ...
    casadi_funs, ...
    X0, ...
    T_real, ...
    20, ...
    1e-5, ...
    1);

%% =========================================================
% RESULTADOS
%% =========================================================

M_opt = full(X_opt_cell{1});

fprintf('\n========================================\n');
fprintf('RESULTADO FINAL\n');
fprintf('========================================\n');

disp('Matriz ótima:');
disp(M_opt);

%% =========================================================
% TESTE DE ORTOGONALIDADE
%% =========================================================

ortho_err = norm(M_opt'*M_opt - eye(n_dim), 'fro');

fprintf('\nErro ortogonalidade: %.3e\n', ortho_err);

%% =========================================================
% TESTE DETERMINANTE
%% =========================================================

detM = det(M_opt);

fprintf('det(M_opt): %.12f\n', detM);

%% =========================================================
% TESTE DAS RESTRIÇÕES
%% =========================================================

g1_opt = -M_opt(n_dim,n_dim);

g2_opt = 1 - trace(M_opt);

g3_opt = cosd(60) - (M_opt*u)'*u;

g_opt = [g1_opt;
         g2_opt;
         g3_opt];

fprintf('\nVALOR DAS RESTRIÇÕES:\n');

for i = 1:length(g_opt)

    fprintf('g(%d) = %+ .12e\n', i, g_opt(i));

end

%% =========================================================
% TESTE DE VIABILIDADE
%% =========================================================

tol_feas = 1e-8;

viol = max(g_opt);

fprintf('\nMaior violação: %.3e\n', viol);

if all(g_opt <= tol_feas)

    fprintf('\nSOLUÇÃO FINAL É VIÁVEL.\n');

else

    fprintf('\nSOLUÇÃO FINAL VIOLA AS RESTRIÇÕES.\n');

end

%% =========================================================
% TESTE EXTRA: POSITIVIDADE DAS FOLGAS
%% =========================================================

s = -g_opt;

fprintf('\nFolgas s = -g(x):\n');
disp(s);

if all(s > 0)

    fprintf('Todas folgas positivas.\n');

else

    fprintf('Existem folgas não positivas.\n');

end