function G_out = run_optimization(sym_data, casadi_funs, G0,param, max_iter, tol,order)
% RUN_OPTIMIZATION
% Executa o processo de otimização usando a parte simbólica pré-compilada
%
% sym_data    : estrutura com dados simbólicos
% casadi_funs : funções CasADi pré-compiladas
% G0          : condição inicial (cell)
% max_iter    : número máximo de iterações
% tol         : tolerância para convergência

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
alpha = 1;

%% =========================================================
%% Extrair funções CasADi
%% =========================================================
f_fun = casadi_funs.f_fun;
gradFun = casadi_funs.gradFun;
HFun = casadi_funs.HFun;

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
    H     = full(HFun(Gvec_num,param));

    if has_g
        g_val = full(g_fun(Gvec_num,param));
        Jg    = full(JgFun(Gvec_num,param));
    else
        g_val = [];
        Jg    = [];
    end

    if has_h
        h_val = full(h_fun(Gvec_num,param));
        Jh    = full(JhFun(Gvec_num,param));
    else
        h_val = [];
        Jh    = [];
    end

    %% ---------------- Residual KKT ----------------
    r_dual = gradL;
    if has_g, r_dual = r_dual + Jg'*nu; end
    if has_h, r_dual = r_dual + Jh'*lambda; end

    r_prim_g = [];
    r_cent   = [];
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

    %% ---------------- Sistema KKT (MODIFICADO APENAS AQUI) ----------------
   %% ---------------- Sistema KKT (CORRIGIDO) ----------------

if has_g && has_h
    
    S = diag(s);
    Nmat = diag(nu);

    KKT = [ H,              Jh',            Jg',            zeros(d*N,m);
            Jh,             zeros(p),       zeros(p,m),     zeros(p,m);
            Jg,             zeros(m,p),     zeros(m),       eye(m);
            zeros(m,d*N),   zeros(m,p),     S,              Nmat ];

    rhs = -[ r_dual;
             r_prim_h;
             r_prim_g;
             r_cent ];

elseif has_g && ~has_h
    
    S = diag(s);
    Nmat = diag(nu);

    KKT = [ H,              Jg',            zeros(d*N,m);
            Jg,             zeros(m),       eye(m);
            zeros(m,d*N),   S,              Nmat ];

    rhs = -[ r_dual;
             r_prim_g;
             r_cent ];

elseif has_h && ~has_g

    KKT = [ H,      Jh';
            Jh,     zeros(p) ];

    rhs = -[ r_dual;
             r_prim_h ];

else

    KKT = H;
    rhs = -r_dual;

end

    %% ---------------- Resolução ----------------
    %% 
    %% 
  reg = 10^-3;
    delta =  ((KKT+reg*eye(size(KKT)))\rhs);

    %% ---------------- Extração ----------------
    idx = 1;
    deta = delta(idx:idx+d*N-1);
    idx = idx + d*N;

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
 alpha = max(0.1,alpha);
    %% ---------------- Atualização geométrica (MANTIDO COMO ESTAVA) ----------------
    j = 1;
    d_max = 2;
    for i = 1:N
        eta_i = alpha * deta(j:j+d-1);
        if(norm(deta(j:j+d-1))>d_max)
         eta_i = alpha*deta(j:j+d-1)/norm(deta(j:j+d-1));        end
        S = full(casadi_funs.S_fun(eta_i));
        Gi_1 = reshape(Gvec_num((i-1)*n^2+1:i*n^2), n, n);
        Gi = Gi_1 * expm(S);

        Gvec_num((i-1)*n^2+1:i*n^2) = Gi(:);
        j = j + d;
    end

    %% ---------------- Atualização dual (MANTIDO COMO ESTAVA) ----------------
    z =[];
    dz = [];
    if has_g
        z = [s;nu]; 
        dz = [ds;dnu];
    end

    if has_h
        z = [z;lambda];
        dz = [dz;dlambda];
    end
     if has_g || has_h
    n_o = length(z);
    omega = zeros(n_o*(n_o-1)/2,1);
    Z = [eye(length(z)), z; zeros(1, length(z)), 1];
    S_z = [skew_number(omega), alpha*dz; zeros(1, length(z)), 0];
    Z = Z * expm(S_z);
     end
    if has_g
        s = Z(1:m, end);
        nu = Z(m+1:2*m, end);
        mu = 0.5*(s'*nu)/max(1,m);
    end
    
    if has_h && ~has_g
        lambda = Z(1:p, end);
    elseif has_h && has_g
        lambda = Z(2*m+1:2*m+p, end);
    end
   if(order ==1)
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