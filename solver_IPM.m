
function [sym_data, casadi_funs] = solver_IPM(f, G_sym, param, ...
                                      S_sym, omega_sym, g, h)

import casadi.*

%% =========================================================
% Dimensoes e flags
%% =========================================================

N = numel(G_sym);
n = size(G_sym{1},1);
d = length(omega_sym);

has_g = ~isempty(g) && numel(g) > 0;
has_h = ~isempty(h) && numel(h) > 0;

m = numel(g);
p = numel(h);

%% =========================================================
% Multiplicadores simbolicos
%% =========================================================

lambda = SX.sym('lambda',p);
nu     = SX.sym('nu',m);

%% =========================================================
% Lagrangiana estendida
%% =========================================================

L = f;

if has_g
    L = L + nu.'*g;
end

if has_h
    L = L + lambda.'*h;
end

%% =========================================================
% Gradiente da funcao objetivo
%% =========================================================

grad_f = symbolic_Jacobian(f, G_sym, S_sym, omega_sym);

%% =========================================================
% Hessiana da Lagrangiana
%% =========================================================

[H_L, H_sym, A_skew] = symbolic_Hessian( ...
    L, G_sym, S_sym, omega_sym, 0);

%% =========================================================
% Jacobianas das restricoes
%% =========================================================

if has_g
    J_g = symbolic_Jacobian(g, G_sym, S_sym, omega_sym);
else
    J_g = SX.zeros(0,d*N);
end

if has_h
    J_h = symbolic_Jacobian(h, G_sym, S_sym, omega_sym);
else
    J_h = SX.zeros(0,d*N);
end

%% =========================================================
% Vetorizacao das variaveis
%% =========================================================

Gvec = [];

for i = 1:N
    Gvec = [Gvec; G_sym{i}(:)];
end

%% =========================================================
% Funcoes CasADi
%% =========================================================

f_fun = Function('f', {Gvec,param}, {f});

gradFun = Function('grad', {Gvec,param}, {grad_f});

% Hessiana da Lagrangiana:
% Entradas: G, parametros, lambda e nu

HFun = Function('H_L', ...
    {Gvec,param,lambda,nu}, {H_L});

H_SymFun = Function('H_L_sym', ...
    {Gvec,param,lambda,nu}, {H_sym});

A_skewFun = Function('A_skew_L', ...
    {Gvec,param,lambda,nu}, {A_skew});

S_fun = Function('S_fun', {omega_sym}, {S_sym});

if has_g

    g_fun = Function('g', {Gvec,param}, {g});

    JgFun = Function('Jg', {Gvec,param}, {J_g});

else

    g_fun = [];
    JgFun = [];

end

if has_h

    h_fun = Function('h', {Gvec,param}, {h});

    JhFun = Function('Jh', {Gvec,param}, {J_h});

else

    h_fun = [];
    JhFun = [];

end

%% =========================================================
% Estrutura de saida
%% =========================================================

sym_data.N = N;
sym_data.n = n;
sym_data.d = d;

sym_data.has_g = has_g;
sym_data.has_h = has_h;

sym_data.m = m;
sym_data.p = p;

casadi_funs.f_fun = f_fun;
casadi_funs.gradFun = gradFun;

casadi_funs.HFun = HFun;
casadi_funs.H_SymFun = H_SymFun;
casadi_funs.A_skewFun = A_skewFun;

casadi_funs.g_fun = g_fun;
casadi_funs.JgFun = JgFun;

casadi_funs.h_fun = h_fun;
casadi_funs.JhFun = JhFun;

casadi_funs.S_fun = S_fun;

end
