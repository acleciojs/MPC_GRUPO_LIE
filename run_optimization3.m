function G_out = run_optimization3(sym_data, casadi_funs, G0,param, max_iter, tol)

import casadi.*

%% =========================================================
%% Helper
%% =========================================================
to_double = @(x) double(full(x));

%% =========================================================
%% Extrair dados
%% =========================================================
N = sym_data.N;
n = sym_data.n;
d = sym_data.d;
has_g = sym_data.has_g;
has_h = sym_data.has_h;
m = sym_data.m;
p = sym_data.p;
has_p = ~isempty(param);
%% =========================================================
%% Funções CasADi
%% =========================================================
gradFun = casadi_funs.gradFun;

if has_g
    g_fun = casadi_funs.g_fun;
    JgFun = casadi_funs.JgFun;
end

if has_h
    h_fun = casadi_funs.h_fun;
    JhFun = casadi_funs.JhFun;
end

%% =========================================================
%% Inicialização
%% =========================================================
if has_p
Gvec_num = [];
for i = 1:N
    Gvec_num = [Gvec_num; G0{i}(:)];
end

%% ? dimensão COMPLETA
nvar = n*N + p + m;

B = eye(nvar);

if has_g
    g_val = to_double(g_fun(Gvec_num,param));
    s  = max(1, -g_val + 1);
    nu = ones(m,1);
else
    s  = [];
    nu = [];
end

if has_h
    lambda = zeros(p,1);
else
    lambda = [];
end

if has_g
    mu = (s'*nu)/max(1,m);
end

%% Gradiente inicial
grad = to_double(gradFun(Gvec_num,param))';

if has_g
    Jg = to_double(JgFun(Gvec_num,param));
    grad = grad + Jg'*nu;
end

if has_h
    Jh = to_double(JhFun(Gvec_num,param));
    grad = grad + Jh'*lambda;
end

gradL_prev = grad;    
    
else
Gvec_num = [];
for i = 1:N
    Gvec_num = [Gvec_num; G0{i}(:)];
end

%% ? dimensão COMPLETA
nvar = n*N + p + m;

B = eye(nvar);

if has_g
    g_val = to_double(g_fun(Gvec_num));
    s  = max(1, -g_val + 1);
    nu = ones(m,1);
else
    s  = [];
    nu = [];
end

if has_h
    lambda = zeros(p,1);
else
    lambda = [];
end

if has_g
    mu = (s'*nu)/max(1,m);
end

%% Gradiente inicial
grad = to_double(gradFun(Gvec_num))';

if has_g
    Jg = to_double(JgFun(Gvec_num));
    grad = grad + Jg'*nu;
end

if has_h
    Jh = to_double(JhFun(Gvec_num));
    grad = grad + Jh'*lambda;
end

gradL_prev = grad;
end
%% =========================================================
%% Loop principal
%% =========================================================
for k = 1:max_iter

    %% ---------- Avaliações ----------
    if(has_p)
      grad = to_double(gradFun(Gvec_num,param))';

    if has_g
        g_val = to_double(g_fun(Gvec_num,param));
        Jg    = to_double(JgFun(Gvec_num,param));
    end

    if has_h
        h_val = to_double(h_fun(Gvec_num,param));
        Jh    = to_double(JhFun(Gvec_num,param));
    end      
        
    else
    grad = to_double(gradFun(Gvec_num,param))';

    if has_g
        g_val = to_double(g_fun(Gvec_num,param));
        Jg    = to_double(JgFun(Gvec_num,param));
    end

    if has_h
        h_val = to_double(h_fun(Gvec_num,param));
        Jh    = to_double(JhFun(Gvec_num,param));
    end
    end
    %% ---------- Gradiente Lagrangiano ----------
    gradL = grad;
    if has_g, gradL = gradL + Jg'*nu; end
    if has_h, gradL = gradL + Jh'*lambda; end

    %% ---------- Resíduos ----------
    r_dual = gradL;

    if has_g
        r_prim_g = g_val + s;
        r_cent   = nu.*s - mu*ones(m,1);
    else
        r_prim_g = [];
        r_cent   = [];
    end

    if has_h
        r_prim_h = h_val;
    else
        r_prim_h = [];
    end

    res_norm = norm([r_dual; r_prim_g; r_prim_h; r_cent]);

    if res_norm < tol
        fprintf('Convergência na iteração %d\n', k);
        break
    end

    %% =====================================================
    %% Hessiana efetiva
    %% =====================================================
    if has_g
        s_safe = max(s, 1e-8);
        W = mu * diag(1./(s_safe.^2));
        B_eff = B + Jg' * W * Jg;
    else
        B_eff = B;
    end

    B_eff = 0.5*(B_eff + B_eff');
    B_eff = B_eff + 1e-8*eye(size(B_eff));

    %% ---------- Gradiente QP ----------
    g_qp = r_dual;

    if has_g
        mu_over_s = mu ./ s_safe;
        g_qp = g_qp ...
            - Jg' * (nu - mu_over_s) ...
            + Jg' * (W * (g_val + s));
    end

    %% ---------- Igualdades ----------
    if has_h
        Aeq = Jh;
        beq = -h_val;
    else
        Aeq = [];
        beq = [];
    end

    %% ---------- Resolver QP ----------
    options = optimoptions('quadprog','Display','off');
    deta_full = quadprog(B_eff, g_qp, [], [], Aeq, beq, [], [], [], options);

    deta_full = to_double(deta_full);

    %% ? Apenas parte primal (Lie)
    deta = deta_full(1:d*N);

    %% ---------- Recuperação ----------
    if has_g
        ds  = -g_val - s - Jg*deta_full;
        dnu = -nu + mu_over_s - W*ds;
        ds  = to_double(ds);
        dnu = to_double(dnu);
    end

    if has_h
        r_qp = B_eff*deta_full + g_qp;
        dlambda = -(Jh*Jh') \ (Jh * r_qp);
        dlambda = to_double(dlambda);
    end

    %% =====================================================
    %% ? Line search (CORRIGIDO)
    %% =====================================================
    alpha = 1;
    tau = 0.99;
    alpha_min = 1e-1;

    if has_g
        
        % ---- s ----
        idx = ds < 0;
        if any(idx)
            ratio = -s(idx) ./ (ds(idx) + 1e-16);
            alpha = min(alpha, tau * min(ratio));
        end

        % ---- nu ----
        idx = dnu < 0;
        if any(idx)
            ratio = -nu(idx) ./ (dnu(idx) + 1e-16);
            alpha = min(alpha, tau * min(ratio));
        end
    end

    % ? evitar alpha zero
    if ~isfinite(alpha) || alpha <= 0
        alpha = alpha_min;
    else
        alpha = max(alpha, alpha_min);
    end

    %% =====================================================
    %% Atualização Lie
    %% =====================================================
    j = 1;
    for i = 1:N
        eta_i = alpha * deta(j:j+d-1);

        S = skew_number(eta_i);
        Gi_1 = reshape(Gvec_num((i-1)*n^2+1:i*n^2), n, n);

        Gi = Gi_1 * expm_numeric(S,100);
        Gvec_num((i-1)*n^2+1:i*n^2) = Gi(:);

        j = j + d;
    end

    %% =====================================================
    %% BFGS
    %% =====================================================
    sk = to_double(alpha * deta_full);

    grad_new = to_double(gradFun(Gvec_num))';

    if has_g
        grad_new = grad_new + Jg'*nu;
    end

    if has_h
        grad_new = grad_new + Jh'*lambda;
    end

    yk = to_double(grad_new - gradL_prev);

    if (sk' * yk) > 1e-12 * norm(sk) * norm(yk)

        Bs = B * sk;

        if (sk' * yk) >= 0.2 * (sk' * Bs)
            theta = 1;
        else
            theta = (0.8 * sk' * Bs) / (sk' * Bs - sk' * yk + 1e-12);
        end

        yk_tilde = theta * yk + (1 - theta) * Bs;

        B = B ...
            - (Bs*Bs') / (sk'*Bs + 1e-12) ...
            + (yk_tilde*yk_tilde') / (yk_tilde'*sk + 1e-12);

        [V,D] = eig((B+B')/2);
        D = diag(D);
        D(D < 1e-6) = 1e-6;
        B = V * diag(D) * V';
    end

    gradL_prev = grad_new;

    if cond(B) > 1e12
        B = eye(nvar);
    end

    %% =====================================================
    %% Atualização dual
    %% =====================================================
    if has_g
        s  = max(s + alpha*ds, 1e-8);
        nu = max(nu + alpha*dnu, 1e-8);
        mu = 0.5*(s'*nu)/max(1,m);
    end

    if has_h
        lambda = lambda + alpha*dlambda;
    end

    fprintf('iter %d |res| = %.2e |alpha| = %.2e\n', k, res_norm, alpha);

end

%% =========================================================
%% Saída
%% =========================================================
G_out = cell(N,1);
for i = 1:N
    G_out{i} = reshape(Gvec_num((i-1)*n^2+1:i*n^2), n, n);
end

end