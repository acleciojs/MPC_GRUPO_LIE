function [sym_data, casadi_funs] = create_symbolic_part_riemanniana_son(f, G_sym, g, h)
% CREATE_SYMBOLIC_PART
% Parte simbólica completa usando:
%   - Jacobiano Riemanniano
%   - Hessiana Riemanniana
%
% f      : escalar SX
% G_sym  : cell {G1,...,GN}, Gi ∈ SL(n)
% g      : desigualdades g(G) <= 0  (SX ou [])
% h      : igualdades   h(G) = 0    (SX ou [])

import casadi.*

%% =========================================================
%% Dimensões
%% =========================================================
N = numel(G_sym);
n = size(G_sym{1},1);
dim = n^2;
dim_total = N*dim;

has_g = ~isempty(g) && numel(g) > 0;
has_h = ~isempty(h) && numel(h) > 0;

m = numel(g);
p = numel(h);

%% =========================================================
%% Gradiente e Hessiana Riemannianos
%% =========================================================
[grad_f,~] = gradiente_riemanniano_son_cell(f, G_sym);
grad_f = grad_f';
H_f = hessiana_riemanniana_son_cell(f, G_sym);

%% =========================================================
%% Jacobianas das restrições (Riemannianas também)
%% =========================================================
if has_g
    J_g = SX.zeros(m, dim_total);
    for i = 1:m
        [grad_gi,~] =  gradiente_riemanniano_son_cell(g(i), G_sym);
        grad_gi = grad_gi';
        J_g(i,:) = grad_gi;
    end
else
    J_g = SX.zeros(0, dim_total);
end

if has_h
    J_h = SX.zeros(p, dim_total);
    for i = 1:p
        [grad_hi,~] = gradiente_riemanniano_son_cell(h(i), G_sym);
        grad_hi = grad_hi';
        J_h(i,:) = grad_hi;
    end
else
    J_h = SX.zeros(0, dim_total);
end

%% =========================================================
%% Vetorização das variáveis
%% =========================================================
Gvec = SX.zeros(dim_total,1);
for i = 1:N
    idx1 = (i-1)*dim + 1;
    idx2 = i*dim;
    Gvec(idx1:idx2) = reshape(G_sym{i}, dim, 1);
end

%% =========================================================
%% Funções CasADi
%% =========================================================
f_fun   = Function('f',    {Gvec}, {f});
gradFun = Function('grad', {Gvec}, {grad_f});
HFun    = Function('H',    {Gvec}, {H_f});

if has_g
    g_fun  = Function('g',  {Gvec}, {g});
    JgFun  = Function('Jg', {Gvec}, {J_g});
else
    g_fun = [];
    JgFun = [];
end

if has_h
    h_fun  = Function('h',  {Gvec}, {h});
    JhFun  = Function('Jh', {Gvec}, {J_h});
else
    h_fun = [];
    JhFun = [];
end

%% =========================================================
%% Estrutura de saída
%% =========================================================
sym_data.N = N;
sym_data.n = n;
sym_data.dim = dim;
sym_data.dim_total = dim_total;
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

end
