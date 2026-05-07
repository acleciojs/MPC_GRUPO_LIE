function G_out = run_optimization2(sym_data, casadi_funs, G0, max_iter, tol)

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

%% =========================================================
%% Funções CasADi
%% =========================================================
gradFun = casadi_funs.gradFun;
HFun    = casadi_funs.HFun;

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
Gvec_num = [];
for i = 1:N
    Gvec_num = [Gvec_num; G0{i}(:)];
end

if has_g
    g_val = full(g_fun(Gvec_num));
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

%% =========================================================
%% Loop principal
%% =========================================================
for k = 1:max_iter

    %% ---------- Avaliações ----------
    gradL = full(gradFun(Gvec_num))';
    H     = full(HFun(Gvec_num));

    if has_g
        g_val = full(g_fun(Gvec_num));
        Jg    = full(JgFun(Gvec_num));
    end

    if has_h
        h_val = full(h_fun(Gvec_num));
        Jh    = full(JhFun(Gvec_num));
    end

    %% ---------- Resíduos ----------
    r_dual = gradL;
    if has_g, r_dual = r_dual + Jg'*nu; end
    if has_h, r_dual = r_dual + Jh'*lambda; end

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
    %% ? SUBPROBLEMA SQP (IPM equivalente)
    %% =====================================================

    if has_g
        s_safe = max(s, 1e-8);
        mu_over_s = mu ./ s_safe;
        W = mu * diag(1./(s_safe.^2));
    end

    %% Hessiana modificada
    B = H;
    if has_g
        B = B + Jg' * W * Jg;
    end

    B = 0.5*(B+B'); % simetrizar

    %% Regularização
    reg = 1e-8;
    B = B + reg*eye(size(B));

    %% Gradiente QP
    g_qp = r_dual;

    if has_g
        g_qp = g_qp ...
            - Jg' * (nu - mu_over_s) ...
            + Jg' * (W * (g_val + s));
    end

    %% Restrições igualdade
    if has_h
        Aeq = Jh;
        beq = -h_val;
    else
        Aeq = [];
        beq = [];
    end

    %% ---------- CHECK NUMÉRICO ----------
    if any(~isfinite(B(:))) || any(~isfinite(g_qp))
        error('B ou g_qp contém NaN/Inf');
    end

    %% ---------- Resolver QP ----------
    options = optimoptions('quadprog','Display','off');
    deta = quadprog(B, g_qp, [], [], Aeq, beq, [], [], [], options);

    %% ---------- Recuperação ----------
    if has_g
        ds  = -g_val - s - Jg*deta;
        dnu = -nu + mu_over_s - W*ds;
    end

    if has_h
        % atualização correta de lambda
        r_qp = B*deta + g_qp;
        dlambda = -(Jh*Jh') \ (Jh * r_qp);
    end

    %% =====================================================
    %% Line search (robusto)
    %% =====================================================
    alpha = 1;
    tau = 0.99;

    if has_g
        if any(ds < 0)
            alpha = min(alpha, tau*min(-s(ds<0)./ds(ds<0)));
        end

        if any(dnu < 0)
            alpha = min(alpha, tau*min(-nu(dnu<0)./dnu(dnu<0)));
        end
    end

    %% =====================================================
    %% Atualização geométrica
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
    %% Atualização dual
    %% =====================================================
    if has_g || has_h

        z = [];
        dz = [];

        if has_g
            z = [s;nu];
            dz = [ds;dnu];
        end

        if has_h
            z = [z;lambda];
            dz = [dz;dlambda];
        end

        n_o = length(z);
        omega = zeros(n_o*(n_o-1)/2,1);

        Z = [eye(n_o), z; zeros(1,n_o), 1];
        S_z = [skew_number(omega), alpha*dz; zeros(1,n_o), 0];

        Z = Z * expm_numeric(S_z,100);

        if has_g
            s  = max(Z(1:m,end), 1e-8);
            nu = max(Z(m+1:2*m,end), 1e-8);
            mu = 0.5*(s'*nu)/max(1,m);
        end

        if has_h && ~has_g
            lambda = Z(1:p,end);
        elseif has_h && has_g
            lambda = Z(2*m+1:2*m+p,end);
        end
    end

    fprintf('iter %d |res| = %.2e |alpha| = %.2f\n', k, res_norm, alpha);

end

%% =========================================================
%% Saída
%% =========================================================
G_out = cell(N,1);
for i = 1:N
    G_out{i} = reshape(Gvec_num((i-1)*n^2+1:i*n^2), n, n);
end

end