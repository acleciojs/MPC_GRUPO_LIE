function [sym_data, casadi_funs] = solver_IPM(f, G_sym,param,S_sym,omega_sym, g, h)
% CREATE_SYMBOLIC_PART
% Cria toda a parte simbólica para otimização em SL(n)
%
% f      : função escalar SX
% G_sym  : cell {G1,...,GN}, Gi ∈ SL(n)
% g      : desigualdades g(G) <= 0 (SX ou [])
% h      : igualdades   h(G) = 0  (SX ou [])
%
% Retorna:
% sym_data  : estrutura com dados simbólicos
% casadi_funs : funções CasADi compiladas

import casadi.*

%% =========================================================
%% Dimensões e flags
%% =========================================================
N = numel(G_sym);
n = size(G_sym{1},1);
d = length(omega_sym);

has_g = ~isempty(g) && numel(g) > 0;
has_h = ~isempty(h) && numel(h) > 0;

m = numel(g);
p = numel(h);

%% =========================================================
%% Gradiente e Hessiana geométricos
%% =========================================================
grad_f = Jacobian_f(f, G_sym,S_sym, omega_sym);
H_f    = Hessian_f(f, G_sym,S_sym, omega_sym);

%% =========================================================
%% Jacobianas das restrições
%% =========================================================
if has_g
    J_g = SX.zeros(m, d*N);
    for i = 1:m
        J_g(i,:) = Jacobian_f(g(i), G_sym,S_sym, omega_sym);
    end
else
    J_g = SX.zeros(0, d*N);
end

if has_h
    J_h = SX.zeros(p, d*N);
    for i = 1:p
        J_h(i,:) = Jacobian_f(h(i), G_sym,S_sym, omega_sym);
    end
else
    J_h = SX.zeros(0, d*N);
end

%% =========================================================
%% Vetorização das variáveis
%% =========================================================
Gvec = [];
for i = 1:N
    Gvec = [Gvec; G_sym{i}(:)];
end

%% =========================================================
%% Funções CasADi
%% =========================================================
f_fun   = Function('f',    {Gvec,param}, {f});
gradFun = Function('grad', {Gvec,param}, {grad_f});
HFun    = Function('H',    {Gvec,param}, {H_f});
S_fun = Function('S_fun', {omega_sym}, {S_sym});
if has_g
    g_fun  = Function('g',  {Gvec,param}, {g});
    JgFun = Function('Jg', {Gvec,param}, {J_g});
else
    g_fun = [];
    JgFun = [];
end

if has_h
    h_fun  = Function('h',  {Gvec,param}, {h});
    JhFun = Function('Jh', {Gvec,param}, {J_h});
else
    h_fun = [];
    JhFun = [];
end

%% =========================================================
%% Estrutura de saída
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
casadi_funs.g_fun = g_fun;
casadi_funs.JgFun = JgFun;
casadi_funs.h_fun = h_fun;
casadi_funs.JhFun = JhFun;
casadi_funs.S_fun = S_fun;
end