function G_out = run_optimization_sqp(sym_data, casadi_funs, G0,param,max_iter, tol)

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
    g_val = full(g_fun(Gvec_num,param));
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
%% Loop principal
%% =========================================================
for k = 1:max_iter

    gradL = full(gradFun(Gvec_num,param))';
    H     = full(HFun(Gvec_num,param));

    if has_g
        g_val = full(g_fun(Gvec_num,param));
        Jg    = full(JgFun(Gvec_num,param));
    end

    if has_h
        h_val = full(h_fun(Gvec_num,param));
        Jh    = full(JhFun(Gvec_num,param));
    end

    %% =====================================================
    %% SQP SUBPROBLEMA (IPM COM LOG)
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

 

    %% -------- Igualdades --------
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

    %% -------- Desigualdade (manter s > 0) --------
    Aineq = [];
    bineq = [];

    if has_g
        tau = 0.995;
        Aineq = [zeros(m,d*N), -eye(m)];
        bineq = tau * s;
    end
 H_bar = 0.5*(H_bar'+H_bar);
 H_bar = H_bar + 1e-8 * eye(size(H_bar));
    %% =====================================================
    %% Resolver QP
    %% =====================================================
    [dx,~,exitflag,~,lambda_struct] = quadprog(H_bar, f_bar, ...
        Aineq, bineq, Aeq, beq, [], [], [], opts);

    if exitflag <= 0
        warning('quadprog falhou, fallback KKT');
        KKT = [H_bar, Aeq';
               Aeq,   zeros(size(Aeq,1))];
        rhs = -[f_bar; beq];
        sol = (KKT + 1e-5*eye(size(KKT)))\rhs;
        dx = sol(1:length(f_bar));
    end

    %% =====================================================
    %% Extrair direção
    %% =====================================================
    if has_g
        deta = dx(1:d*N);
        ds   = dx(d*N+1:d*N+m);
    else
        deta = dx;
    end

    %% Recuperar duais (opcional)
    if has_g
        nu = lambda_struct.eqlin(1:m);
    end

    if has_h
        if has_g
            lambda = lambda_struct.eqlin(m+1:end);
        else
            lambda = lambda_struct.eqlin;
        end
    end

    %% =====================================================
    %% Line search (positividade)
    %% =====================================================
    alpha = 1;

    if has_g
        idx = ds < 0;
        if any(idx)
            alpha = min(alpha, min(-s(idx)./ds(idx)));
        end
    end

    %% =====================================================
    %% Atualização geométrica
    %% =====================================================
    j = 1;
    d_max = 5;

    for i = 1:N
        eta_i = alpha * deta(j:j+d-1);

        if norm(eta_i) > d_max
            eta_i = d_max * eta_i / norm(eta_i);
        end

        S = full(casadi_funs.S_fun(eta_i));
        Gi_1 = reshape(Gvec_num((i-1)*n^2+1:i*n^2), n, n);
        Gi = Gi_1 * expm(S);

        Gvec_num((i-1)*n^2+1:i*n^2) = Gi(:);
        j = j + d;
    end

    %% =====================================================
    %% Atualizar slack
    %% =====================================================
    if has_g
        
        s = s + alpha*ds;
        s = max(s, 1e-3);
     
    end

    %% =====================================================
    %% Atualizar ?
    %% =====================================================
    if has_g
        mu = 0.2 * (s' * nu) / max(1,m);
    end

    %% =====================================================
    %% Critério de parada simples
    %% =====================================================
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