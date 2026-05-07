function G_out = run_optimization_sqp_filter(sym_data, casadi_funs, G0, max_iter, tol)

import casadi.*

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
f_fun   = casadi_funs.f_fun;
gradFun = casadi_funs.gradFun;
HFun    = casadi_funs.HFun;

if has_g
    g_fun  = casadi_funs.g_fun;
    JgFun  = casadi_funs.JgFun;
end

if has_h
    h_fun  = casadi_funs.h_fun;
    JhFun  = casadi_funs.JhFun;
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

opts = optimoptions('quadprog','Display','off');

%% =========================================================
%% FILTRO (inicialização)
%% =========================================================
filter = [];
gamma_theta = 1e-4;
gamma_f = 1e-4;
beta = 0.5;

%% =========================================================
%% Loop principal
%% =========================================================
for k = 1:max_iter

    gradL = full(gradFun(Gvec_num))';
    H     = full(HFun(Gvec_num));

    if has_g
        g_val = full(g_fun(Gvec_num));
        Jg    = full(JgFun(Gvec_num));
        theta = norm(g_val,1);
    else
        theta = 0;
    end

    if has_h
        h_val = full(h_fun(Gvec_num));
        Jh    = full(JhFun(Gvec_num));
    end

    f_val = full(f_fun(Gvec_num));

    %% =====================================================
    %% SQP SUBPROBLEMA
    %% =====================================================
    if has_g
        invS = diag(1./s);
        Hs = mu * (invS.^2);
        H_bar = blkdiag(H, Hs);

        f_bar = [gradL;
                 -mu*(1./s)];
    else
        H_bar = H;
        f_bar = gradL;
    end

    H_bar = H_bar + 1e-8 * eye(size(H_bar));

    %% Igualdades
    Aeq = [];
    beq = [];

    if has_g
        Aeq = [Aeq; [Jg, eye(m)]];
        beq = [beq; -(g_val + s)];
    end

    if has_h
        if has_g
            Aeq = [Aeq; [Jh, zeros(p,m)]];
        else
            Aeq = [Aeq; Jh];
        end
        beq = [beq; -h_val];
    end

    %% Desigualdade (s > 0)
    Aineq = [];
    bineq = [];

    if has_g
        tau = 0.995;
        Aineq = [zeros(m,d*N), -eye(m)];
        bineq = tau * s;
    end

    H_bar = 0.5*(H_bar'+H_bar);

    %% Resolver QP
    [dx,~,exitflag,~,lambda_struct] = quadprog(H_bar, f_bar, ...
        Aineq, bineq, Aeq, beq, [], [], [], opts);

    if exitflag <= 0
       % warning('quadprog falhou, fallback KKT');
        KKT = [H_bar, Aeq';
               Aeq,   zeros(size(Aeq,1))];
        rhs = -[f_bar; beq];
        sol = (KKT + 1e-8*eye(size(KKT)))\rhs;
        dx = sol(1:length(f_bar));
    end

    %% Extrair direção
    if has_g
        deta = dx(1:d*N);
        ds   = dx(d*N+1:d*N+m);
    else
        deta = dx;
    end

    %% =====================================================
    %% LINE SEARCH COM FILTRO
    %% =====================================================
    alpha = 1;

    if has_g
        idx = ds < 0;
        if any(idx)
            alpha = min(alpha, 0.99 * min(-s(idx)./ds(idx)));
        end
    end

    accept = false;
    d_max = 5;

    while ~accept

        G_trial = Gvec_num;
        s_trial = s;

        % atualização geométrica (trial)
        j = 1;
        for i = 1:N
            eta_i = alpha * deta(j:j+d-1);

            if norm(eta_i) > d_max
                eta_i = d_max * eta_i / norm(eta_i);
            end

            S = skew_number2(eta_i);
            Gi_1 = reshape(G_trial((i-1)*n^2+1:i*n^2), n, n);
            Gi = Gi_1 * expm(S);

            G_trial((i-1)*n^2+1:i*n^2) = Gi(:);
            j = j + d;
        end

        if has_g
            s_trial = s + alpha*ds;
            s_trial = max(s_trial, 1e-8);
            g_trial = full(g_fun(G_trial));
            theta_trial = norm(g_trial,1);
        else
            theta_trial = 0;
        end

        f_trial = full(f_fun(G_trial));

        %% filtro
        accept = true;
        for i_f = 1:length(filter)
            if (theta_trial >= (1 - gamma_theta)*filter(i_f).theta) && ...
               (f_trial >= filter(i_f).f - gamma_f*filter(i_f).theta)
                accept = false;
                break;
            end
        end

        if ~accept
            alpha = beta * alpha;
            if alpha < 1e-8
               % warning('alpha muito pequeno (filtro)');
                break;
            end
        end
    end

    %% =====================================================
    %% ATUALIZAÇÃO REAL
    %% =====================================================
    j = 1;
    for i = 1:N
        eta_i = alpha * deta(j:j+d-1);

        if norm(eta_i) > d_max
            eta_i = d_max * eta_i / norm(eta_i);
        end

        S = skew_number2(eta_i);
        Gi_1 = reshape(Gvec_num((i-1)*n^2+1:i*n^2), n, n);
        Gi = Gi_1 * expm(S);

        Gvec_num((i-1)*n^2+1:i*n^2) = Gi(:);
        j = j + d;
    end

    if has_g
        s = s + alpha*ds;
        s = max(s, 1e-3);
    end

    %% =====================================================
    %% ATUALIZAR FILTRO
    %% =====================================================
    if has_g
        theta_new = norm(full(g_fun(Gvec_num)),1);
    else
        theta_new = 0;
    end

    f_new = full(f_fun(Gvec_num));

    % remover dominados
    to_keep = true(length(filter),1);
    for i_f = 1:length(filter)
        if (theta_new <= filter(i_f).theta) && (f_new <= filter(i_f).f)
            to_keep(i_f) = false;
        end
    end
    filter = filter(to_keep);

    filter(end+1).theta = theta_new;
    filter(end).f = f_new;

    %% =====================================================
    %% Atualizar ?
    %% =====================================================
    if has_g
        mu = 0.2 * (s' * nu) / max(1,m);
    end

    %% Critério de parada
    if norm(deta) < tol
        fprintf('Convergência na iteração %d\n', k);
        break
    end

end

%% =========================================================
%% Saída
%% =========================================================
G_out = cell(N,1);
for i = 1:N
    G_out{i} = reshape(Gvec_num((i-1)*n^2+1:i*n^2), n, n);
end

end