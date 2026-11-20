function [H_sym, grad] = symbolic_hessian_simetrica_pullback( ...
    f,G_list,S_sym,omega_sym,expm_mode)
% SYMBOLIC_HESSIAN_SIMETRICA_PULLBACK
% Retorna SOMENTE a Hessiana simetrica e o gradiente do pullback por
% perturbacoes a direita:
%
%   f_pert(xi) = f(G_1 * exp(S(xi_1)),...,G_N * exp(S(xi_N))).
%
%   grad  = d f_pert / d xi |_(xi=0)          (m*N x 1)
%   H_sym = d2 f_pert / d xi2 |_(xi=0)        (m*N x m*N)
%
% NAO calcula A_skew, comutadores, pinv_E ou L2 completo.
%
% expm_mode='jet2' (padrao): para calcular DERIVADAS ATE ORDEM 2 no
% ponto xi=0, exp(S) pode ser substituida por I + S + 0.5*S*S.
% A diferenca e O(||xi||^3), portanto o gradiente e a Hessiana em
% xi=0 sao EXATAMENTE iguais aos da exponencial, para f suave.
% Esse modo evita a geracao da exponencial simbolica completa.
%
% expm_mode='expm': usa a funcao externa expm_casadi(S), como no
% symbolic_Hessian original. Mais caro; util para verificacao.
%
% Cada G_list{r} tem as mesmas coordenadas de S_sym. G_list pode conter
% simbolos SX ou MX. A ordem de xi e [xi_1;...;xi_N].

import casadi.*

if nargin < 5 || isempty(expm_mode)
    expm_mode = 'jet2';
end
assert(iscell(G_list) && ~isempty(G_list), ...
    'G_list deve ser uma celula nao vazia.');
assert(numel(f) == 1, 'A funcao f deve ser escalar.');

N = numel(G_list);
m = numel(omega_sym);
total_vars = m*N;

if isa(f,'casadi.SX')
    xi = SX.sym('xi_pullback',total_vars,1);
elseif isa(f,'casadi.MX')
    xi = MX.sym('xi_pullback',total_vars,1);
else
    error('f deve ser uma expressao CasADi SX ou MX.');
end

G_pert_cell = cell(N,1);
for r = 1:N
    ii = (r-1)*m + (1:m);
    S_r = substitute(S_sym,omega_sym,xi(ii));
    n = size(G_list{r},1);
    assert(size(G_list{r},2)==n && isequal(size(S_r),[n,n]), ...
        'A dimensao de S deve coincidir com a de cada matriz G.');

    switch lower(expm_mode)
        case 'jet2'
            % Segundo jato da exponencial: Hessiana EXATA em xi=0.
            E_r = eye(n) + S_r + 0.5*(S_r*S_r);
        case 'expm'
            assert(exist('expm_casadi','file')==2, ...
                'expm_casadi.m nao esta no path MATLAB.');
            E_r = expm_casadi(S_r);
        otherwise
            error('expm_mode invalido: use jet2 ou expm.');
    end
    G_pert_cell{r} = G_list{r}*E_r;
end

% A substituicao em lote evita substituir parcialmente a mesma expressao.
aux = substitute({f},G_list,G_pert_cell);
f_pert = aux{1};

% Apenas a derivacao simetrica do pullback (tip=1 da funcao original).
H_expr = hessian(f_pert,xi);
g_expr = jacobian(f_pert,xi).';
xi0 = zeros(total_vars,1);
H_raw = substitute(H_expr,xi,xi0);
grad = substitute(g_expr,xi,xi0);

% A Hessiana escalar e simetrica; esta linha ajuda na aritmetica numerica.
H_sym = 0.5*(H_raw+H_raw.');
end
