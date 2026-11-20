function G_out = run_optimization_L2(sym_data, casadi_funs, G0, param, max_iter, tol, order)
% RUN_OPTIMIZATION_L2 com Não-Cancelamento e Gauge Fixing

%% =========================================================
%% Extrair dados da estrutura simbólica
%% =========================================================
N = sym_data.N;
n = sym_data.n;
d = sym_data.d;
has_g = sym_data.has_g;
has_h = sym_data.has_h;
m = sym_data.m;
p = sym_data.p;

n_prim = d * N;

%% =========================================================
%% Extrair funções CasADi
%% =========================================================
gradFun  = casadi_funs.gradFun;
H_SymFun = casadi_funs.H_SymFun;
A_skewFun= casadi_funs.A_skewFun;
if has_g
    g_fun = casadi_funs.g_fun;
    JgFun = casadi_funs.JgFun;
end
if has_h
    h_fun = casadi_funs.h_fun;
    JhFun = casadi_funs.JhFun;
end

%% =========================================================
%% Inicialização numérica
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

%% =========================================================
%% Loop principal
%% =========================================================
for k = 1:max_iter

    gradL = full(gradFun(Gvec_num,param))';
    
    H = full(H_SymFun(Gvec_num,param));
    H = 0.5 * (H + H.'); % Força simetria exata
    
    A = full(A_skewFun(Gvec_num,param));

    if has_g
        g_val = full(g_fun(Gvec_num,param));
        Jg    = full(JgFun(Gvec_num,param));
    else
        g_val = []; Jg = [];
    end

    if has_h
        h_val = full(h_fun(Gvec_num,param));
        Jh    = full(JhFun(Gvec_num,param));
    else
        h_val = []; Jh = [];
    end

    %% ---------------- Residual KKT ----------------
    r_dual = gradL;
    if has_g, r_dual = r_dual + Jg'*nu; end
    if has_h, r_dual = r_dual + Jh'*lambda; end

    r_prim_g = []; r_cent = [];
    if has_g
        r_prim_g = g_val + s;
        r_cent   = nu.*s - mu*ones(m,1);
    end

    r_prim_h = [];
    if has_h
        r_prim_h = h_val;
    end

    res_norm = norm([r_dual; r_prim_g; r_prim_h; r_cent]);

    if res_norm < tol
        fprintf('Convergência na iteração %d\n', k);
        break
    end

    %% ---------------- Sistema KKT com Não-Cancelamento ----------------
    % Parâmetros de estabilização
    reg = 1e-2;
    mu_gauge     = abs(min(eig(H)))+reg;  % Quebra a redundância [v;-v]
    rho_barreira = 1e-8;  % Força os deltas a não se cancelarem (||δ1+δ2|| > 0)
    
    I_g = eye(n_prim);
    
    % Bloco primal com as duas penalizações:
    % A diagonal recebe mu_gauge (estabilidade) + rho_barreira (não-cancelamento)
    H2 = [H + (mu_gauge + rho_barreira)*I_g,   H + A; 
          H - A,                H + (mu_gauge + rho_barreira)*I_g];

    if has_g && has_h
        S = diag(s);
        Nmat = diag(nu);
        
        Jh2 = [Jh, Jh];
        Jg2 = [Jg, Jg];

        KKT = [ H2,              Jh2',          Jg2',          zeros(2*n_prim, m);
                Jh2,             zeros(p),       zeros(p,m),    zeros(p,m);
                Jg2,             zeros(m,p),     zeros(m),      eye(m);
                zeros(m,2*n_prim), zeros(m,p),   S,             Nmat ];

        rhs = -[ r_dual; 
                 r_dual; 
                 r_prim_h; 
                 r_prim_g; 
                 r_cent ];

    elseif has_g && ~has_h
        S = diag(s);
        Nmat = diag(nu);
        Jg2 = [Jg, Jg];

        KKT = [ H2,              Jg2',          zeros(2*n_prim, m);
                Jg2,             zeros(m),       eye(m);
                zeros(m,2*n_prim), S,            Nmat ];

        rhs = -[ r_dual; 
                 r_dual; 
                 r_prim_g; 
                 r_cent ];

    elseif has_h && ~has_g
        Jh2 = [Jh, Jh];

        KKT = [ H2,              Jh2';
                Jh2,             zeros(p) ];

        rhs = -[ r_dual; 
                 r_dual; 
                 r_prim_h ];

    else
        KKT = H2;
        rhs = -[ r_dual; r_dual ];
    end

    %% ---------------- Resolução com LU Adaptativo ----------------
    delta = solve_KKT(KKT, rhs);

    %% ---------------- Extração ----------------
    idx = 1;
    deta1 = delta(idx:idx+n_prim-1);
    idx = idx + n_prim;
    
    deta2 = delta(idx:idx+n_prim-1);
    idx = idx + n_prim;

    if has_h
        dlambda = delta(idx:idx+p-1);
        idx = idx + p;
    end

    if has_g
        dnu = delta(idx:idx+m-1);
        idx = idx + m;
        ds = delta(idx:end);
    end

    %% ---------------- Line search ----------------
    alpha = 1;

    if has_g
        idxs = ds < 0;
        if any(idxs)
            alpha = min(alpha, min(-s(idxs)./ds(idxs)));
        end
        idxn = dnu < 0;
        if any(idxn)
            alpha = min(alpha, min(-nu(idxn)./dnu(idxn)));
        end
    end
    alpha = 1 * alpha; % Margem de segurança

    %% ---------------- Atualização geométrica ----------------
    j = 1;
    for i = 1:N
        step1 = deta1(j:j+d-1);
        step2 = deta2(j:j+d-1);
        
        eta1 = alpha * step1;
        eta2 = alpha * step2;
        
        S1 = full(casadi_funs.S_fun(eta1));
        S2 = full(casadi_funs.S_fun(eta2));
        
        % Ordem correta da formulação BCH
        Gi_1 = reshape(Gvec_num((i-1)*n^2+1:i*n^2), n, n);
        Gi = Gi_1 * expm(S1) * expm(S2);

        Gvec_num((i-1)*n^2+1:i*n^2) = Gi(:);
        j = j + d;
    end

    %% ---------------- Atualização dual (LINEAR) ----------------
    if has_g
        s  = s  + alpha * ds;
        nu = nu + alpha * dnu;
    end
    if has_h
        lambda = lambda + alpha * dlambda;
    end

    % Redução ativa de mu (parâmetro de barreira)
    if has_g
        sigma = 0.1;
        mu = sigma * (s'*nu) / max(1,m);
        
        % Reset de segurança
        if any(s < 1e-10), s = max(1e-3, -g_val + 1); end
        if any(nu < 1e-10), nu = ones(m,1); end
    end

    if order == 1
        fprintf('iter %d |res| = %.2e |alpha| = %.2f\n', k, res_norm, alpha);
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

%% =========================================================
%% SOLVER LU ADAPTATIVO
%% =========================================================
function delta = solve_KKT(K, rhs)
    n_total = size(K, 1);
    escala = max(norm(K, 'fro') / sqrt(n_total), 1);
    lista_reg = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1,1,10,1000] * escala;
    
    for reg = lista_reg
        K_reg = K + reg * eye(n_total);
        [L, U, P] = lu(K_reg);
        if rcond(full(U)) > 1e-14
            delta = U \ (L \ (P * rhs));
            return
        end
    end
    
    warning('KKT singular; usando mínimos quadrados.');
    delta = (K' * K + 1e-4 * escala * eye(n_total)) \ (K' * rhs);
end