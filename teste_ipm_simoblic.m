clear; clc;
import casadi.*

%% =========================================================
%% Dimensões
%% =========================================================
n = 3;
N = 1;

%% =========================================================
%% Variáveis simbólicas G ? SO(3)
%% =========================================================
G = SX.sym('G', n, n);
G_sym = {G};

%% =========================================================
%% Parâmetros (exemplo)
%% =========================================================
param = SX.sym('p',1);

%% =========================================================
%% Variável da álgebra
%% =========================================================
omega = SX.sym('omega',3);

%% =========================================================
%% Mapeamento S(omega) ? so(3)
%% =========================================================
S_sym = [     0      -omega(3)  omega(2);
           omega(3)     0      -omega(1);
          -omega(2)  omega(1)     0 ];

%% =========================================================
%% Função objetivo
%% Exemplo: minimizar distância até identidade
%% =========================================================
f = sumsqr(G - SX.eye(3));

%% =========================================================
%% Restrições (opcional)
%% =========================================================

% desigualdade: exemplo simples
g = [G(1,1) - 0.5];   % <= 0

% igualdade: manter determinante ? 1 (não necessário em SO, mas teste)
h = [det(G) - 1];

%% =========================================================
%% Criar solver simbólico
%% =========================================================
[sym_data, casadi_funs] = solver_IPM(f, G_sym, param, S_sym, omega, g, h);

disp('Estrutura simbólica:');
disp(sym_data);

%% =========================================================
%% Teste numérico
%% =========================================================

% valor de G (identidade)
G_val = eye(3);

% vetorizar
Gvec_val = G_val(:);

% parâmetro
p_val = 0;

%% =========================================================
%% Avaliar funções
%% =========================================================

f_val = full(casadi_funs.f_fun(Gvec_val, p_val));
grad_val = full(casadi_funs.gradFun(Gvec_val, p_val));
H_val = full(casadi_funs.HFun(Gvec_val, p_val));

disp('f(G):');
disp(f_val);

disp('grad f:');
disp(grad_val);

disp('Hessiana:');
disp(H_val);

%% =========================================================
%% Testar restrições
%% =========================================================
if sym_data.has_g
    g_val = full(casadi_funs.g_fun(Gvec_val, p_val));
    Jg_val = full(casadi_funs.JgFun(Gvec_val, p_val));

    disp('g(G):');
    disp(g_val);

    disp('Jg:');
    disp(Jg_val);
end

if sym_data.has_h
    h_val = full(casadi_funs.h_fun(Gvec_val, p_val));
    Jh_val = full(casadi_funs.JhFun(Gvec_val, p_val));

    disp('h(G):');
    disp(h_val);

    disp('Jh:');
    disp(Jh_val);
end

%% =========================================================
%% Testar S_fun
%% =========================================================
omega_test = [1; 2; 3];

S_val = full(casadi_funs.S_fun(omega_test));

disp('S(omega):');
disp(S_val);