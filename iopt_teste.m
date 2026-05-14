clc;
clear;
import casadi.*

%% =========================================================
% Problema:
%
% min (x-1)^2 + (y-2)^2
% s.a.
% x^2 + y^2 = 1
%% =========================================================

%% Variáveis
x = SX.sym('x');
y = SX.sym('y');

X = [x;y];

%% Função objetivo
f = (x-1)^2 + (y-2)^2;

%% Restrição
g = x^2 + y^2 - 1+sin(x);

%% Lagrangiano
lambda = SX.sym('lambda');

L = f + lambda*g;

%% Gradiente do Lagrangiano
gradL = gradient(L, X);

%% Funções CasADi
g_fun     = Function('g_fun',{X},{g});
gradf_fun = Function('gradf_fun',{X},{gradient(f,X)});
gradL_fun = Function('gradL_fun',{X,lambda},{gradL});

%% NLP
nlp = struct('x', X, ...
             'f', f, ...
             'g', g);

%% IPOPT options
opts = struct;
opts.ipopt.print_level = 0;
opts.print_time = false;

solver = nlpsol('solver','ipopt',nlp,opts);

%% =========================================================
% Testes aleatórios
%% =========================================================

N = 10;

for k = 1:N

    fprintf('\n============================\n');
    fprintf('Teste %d\n',k);

    %% Condição inicial aleatória
    x0 = randn(2,1);

    %% Resolve NLP
    sol = solver( ...
        'x0', x0, ...
        'lbg', 0, ...
        'ubg', 0);

    %% Solução
    Xsol = full(sol.x);

    x_sol = Xsol(1);
    y_sol = Xsol(2);

    %% Multiplicador
    lambda_sol = full(sol.lam_g);

    %% Resíduo primal (restrição)
    r_primal = full(g_fun(Xsol));

    %% Resíduo dual
    r_dual = full(gradL_fun(Xsol, lambda_sol));

    %% Norma KKT
    kkt_norm = norm([r_dual; r_primal]);

    %% Prints
    fprintf('x0 = [%+.4f  %+.4f]\n', x0(1), x0(2));

    fprintf('sol = [%+.8f  %+.8f]\n', ...
            x_sol, y_sol);

    fprintf('f(x) = %.10f\n', full(sol.f));

    fprintf('\nResíduo primal:\n');
    fprintf('|g(x)| = %.3e\n', abs(r_primal));

    fprintf('\nResíduo dual:\n');
    fprintf('||grad L|| = %.3e\n', norm(r_dual));

    fprintf('\nNorma KKT total:\n');
    fprintf('||KKT|| = %.3e\n', kkt_norm);

end