function G_out = run_sqp(sym_data, casadi_funs, G0, param, max_iter, tol, order)
% SQP com duas exponenciais. H2 é SIMÉTRICA e preserva B.
% Se H2 for indefinida, adiciona regularização adaptativa.

import casadi.*

%% Extrair dados
N = sym_data.N; n = sym_data.n; d = sym_data.d;
has_g = sym_data.has_g; has_h = sym_data.has_h;
m = sym_data.m; p = sym_data.p;
dN = d*N;

gradFun   = casadi_funs.gradFun;
H_SymFun  = casadi_funs.H_SymFun;
A_skewFun = casadi_funs.A_skewFun;
f_fun     = casadi_funs.f_fun;
if has_g, g_fun = casadi_funs.g_fun; JgFun = casadi_funs.JgFun; end
if has_h, h_fun = casadi_funs.h_fun; JhFun = casadi_funs.JhFun; end

%% Inicialização
Gvec_num = [];
for i = 1:N, Gvec_num = [Gvec_num; G0{i}(:)]; end
if has_h, lambda = zeros(p,1); else, lambda = []; end
if has_g, nu = ones(m,1); else, nu = []; end

%% Parâmetros
mu_g = 1e-4; rho_b = 1e-3;
rho = 100; c_1 = 1e-4; alpha_min = 1e-6;
reg_min = 1e-6;   % regularização mínima para PSD

%% Loop principal
for k = 1:max_iter

    % Avaliar
    gradL = full(gradFun(Gvec_num, param))';
    G     = full(H_SymFun(Gvec_num, param));  G = 0.5*(G + G.');
    B     = full(A_skewFun(Gvec_num, param)); B = 0.5*(B - B.');
    if has_h
        h_val = full(h_fun(Gvec_num, param)); Jh = full(JhFun(Gvec_num, param));
    else, h_val = []; Jh = []; end
    if has_g
        g_val = full(g_fun(Gvec_num, param)); Jg = full(JgFun(Gvec_num, param));
    else, g_val = []; Jg = []; end

    % Resíduo KKT
    r_dual = gradL;
    if has_h, r_dual = r_dual + Jh'*lambda; end
    if has_g, r_dual = r_dual + Jg'*nu; end
    r_prim_h = h_val;
    r_prim_g = [];
    if has_g, r_prim_g = max(0, g_val); end
    res_norm = norm([r_dual; r_prim_h; r_prim_g]);

    if order == 1
        fprintf('[iter %d] res=%.3e  ||B||=%.2e  min_eig(G)=%.2e\n', ...
                k, res_norm, norm(B,'fro'), min(eig(G)));
    end
    if res_norm < tol, fprintf('Convergiu em %d iterações.\n', k); break; end

    % ---- H2 com B (SIMÉTRICA) ----
    I = eye(dN);
    H2 = [ G + (mu_g + rho_b)*I ,  G + B - mu_g*I ;
           G - B - mu_g*I        ,  G + (mu_g + rho_b)*I ];

    % Verificar simetria (deve ser ~0)
    asym = norm(H2 - H2', 'fro') / max(norm(H2,'fro'), 1);
    if order == 1 && asym > 1e-10
        fprintf('  [ALERTA] H2 não simétrica (asym=%.3e). Verifique!\n', asym);
    end

    % ---- Regularização adaptativa para garantir PSD ----
    min_eig_H2 = min(eig(H2));
    if min_eig_H2 < reg_min
        eta_reg = reg_min - min_eig_H2;
        H2_reg  = H2 + eta_reg * eye(2*dN);
        if order == 1
            fprintf('  H2 regularizada: +%.2e I (era %.2e)\n', eta_reg, min_eig_H2);
        end
    else
        H2_reg = H2;
    end

    % ---- Gradiente empilhado ----
    grad2 = [gradL; gradL];

    % ---- Restrições ----
    if has_h, Aeq = [Jh, Jh]; beq = -h_val; else, Aeq = []; beq = []; end
    if has_g, Aineq = [Jg, Jg]; bineq = -g_val; else, Aineq = []; bineq = []; end

    % ---- Resolver QP (H2 é simétrica, aceita qpoases) ----
    [delta, mu_hat, lam_hat] = solve_qp_qpoases(H2_reg, grad2, Aeq, beq, Aineq, bineq, 2*dN, p, m);

    delta1 = delta(1:dN);
    delta2 = delta(dN+1:2*dN);

    % ---- Line search (mérito l1) ----
    phi_0 = full(f_fun(Gvec_num, param)) + rho*(norm(h_val,1) + norm(max(0,g_val),1));
    D_phi = gradL'*delta1 + gradL'*delta2 - rho*(norm(h_val,1) + norm(max(0,g_val),1));

    alpha = 1;
    for ls = 1:30
        Gvec_try = update_Gvec_2exp(Gvec_num, alpha*delta1, alpha*delta2, N, n, d, casadi_funs.S_fun);
        f_try = full(f_fun(Gvec_try, param));
        h_try = []; g_try = [];
        if has_h, h_try = full(h_fun(Gvec_try, param)); end
        if has_g, g_try = full(g_fun(Gvec_try, param)); end
        phi_try = f_try + rho*(norm(h_try,1) + norm(max(0,g_try),1));
        if phi_try <= phi_0 + c_1*alpha*D_phi, break; end
        alpha = alpha/2;
        if alpha < alpha_min, break; end
    end

    % ---- Atualizar ----
    Gvec_num = update_Gvec_2exp(Gvec_num, alpha*delta1, alpha*delta2, N, n, d, casadi_funs.S_fun);
    if has_h, lambda = lambda + alpha*mu_hat; end
    if has_g, nu = max(0, nu + alpha*lam_hat); end

    if order == 1
        fprintf('[iter %d] res=%.2e  alpha=%.2f  ||δ||=%.2e\n', ...
                k, res_norm, alpha, norm(delta));
    end
end

G_out = cell(N,1);
for i = 1:N, G_out{i} = reshape(Gvec_num((i-1)*n^2+1:i*n^2), n, n); end
end

%% =========================================================
%% QP solver via qpoases
%% =========================================================
function [delta, mu_hat, lam_hat] = solve_qp_qpoases(H, grad, Aeq, beq, Aineq, bineq, nv, p, m)
    import casadi.*
    x = SX.sym('x_qp', nv);
    f = 0.5*x'*H*x + grad'*x;

    g_list = {}; lbg = []; ubg = [];
    if ~isempty(Aeq)
        g_list{end+1} = Aeq*x;
        lbg = [lbg; beq]; ubg = [ubg; beq];
    end
    if ~isempty(Aineq)
        g_list{end+1} = Aineq*x;
        lbg = [lbg; -inf(m,1)]; ubg = [ubg; bineq];
    end

    if isempty(g_list)
        qp = struct('x', x, 'f', f);
    else
        qp = struct('x', x, 'f', f, 'g', vertcat(g_list{:}));
    end
    solver = qpsol('qp_solver', 'qpoases', qp, struct('printLevel', 'none'));
    if isempty(g_list)
        sol = solver('lbx', -inf(nv,1), 'ubx', inf(nv,1));
    else
        sol = solver('lbg', lbg, 'ubg', ubg, 'lbx', -inf(nv,1), 'ubx', inf(nv,1));
    end
    delta = full(sol.x);
    if ~isempty(g_list)
        lam_g = full(sol.lam_g);
        if ~isempty(Aeq) && ~isempty(Aineq)
            mu_hat = lam_g(1:p); lam_hat = lam_g(p+1:end);
        elseif ~isempty(Aeq)
            mu_hat = lam_g; lam_hat = [];
        else
            mu_hat = []; lam_hat = lam_g;
        end
    else
        mu_hat = []; lam_hat = [];
    end
end

%% =========================================================
%% Atualização do Gvec
%% =========================================================
function Gvec_new = update_Gvec_2exp(Gvec, delta1, delta2, N, n, d, S_fun)
    Gvec_new = Gvec;
    for i = 1:N
        idx  = (i-1)*d + (1:d);
        S1_i = full(S_fun(delta1(idx)));
        S2_i = full(S_fun(delta2(idx)));
        G_i  = reshape(Gvec((i-1)*n^2+1:i*n^2), n, n);
        G_i  = G_i * expm(S1_i) * expm(S2_i);
        Gvec_new((i-1)*n^2+1:i*n^2) = G_i(:);
    end
end