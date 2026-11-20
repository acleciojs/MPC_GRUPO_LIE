
function G_out = run_optimization(sym_data, casadi_funs, G0, param, max_iter, tol, order,reg)
% RUN_OPTIMIZATION
% Metodo de pontos interiores em SL(n)
% Utiliza a Hessiana geometrica da Lagrangiana estendida.
%
% L(G,lambda,nu) = f(G) + lambda'*h(G) + nu'*g(G)
%
% lambda : multiplicadores das igualdades
% nu     : multiplicadores das desigualdades
%
% KKT utiliza:
% H_L = Hessiana geometrica da Lagrangiana estendida.

%% =========================================================
% Extrair dados da estrutura simbolica
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
% Extrair funcoes CasADi
%% =========================================================

f_fun = casadi_funs.f_fun;
gradFun = casadi_funs.gradFun;

HFun = casadi_funs.HFun;
H_SymFun = casadi_funs.H_SymFun;
A_skewFun = casadi_funs.A_skewFun;

if has_g
    g_fun = casadi_funs.g_fun;
    JgFun = casadi_funs.JgFun;
end

if has_h
    h_fun = casadi_funs.h_fun;
    JhFun = casadi_funs.JhFun;
end

%% =========================================================
% Inicializacao numerica
%% =========================================================

Gvec_num = [];

for i = 1:N
    Gvec_num = [Gvec_num; G0{i}(:)];
end

% Multiplicadores de igualdade
if has_h
    lambda = zeros(p,1);
else
    lambda = zeros(0,1);
end

% Multiplicadores de desigualdade
if has_g

    g_val = full(g_fun(Gvec_num,param));

    s = max(1, -g_val + 1);

    nu = ones(m,1);

    mu = (s'*nu)/max(1,m);

else

    s = zeros(0,1);
    nu = zeros(0,1);

    mu = 0;

end

%% =========================================================
% Loop principal
%% =========================================================

for k = 1:max_iter

    %% =====================================================
    % Gradiente da Lagrangiana
    %% =====================================================

    gradL = full(gradFun(Gvec_num,param))';

    %% =====================================================
    % HESSIANA DA LAGRANGIANA ESTENDIDA
    %% =====================================================

    % H_L = H_f + sum(nu_i H_gi) + sum(lambda_j H_hj)

    A = full(HFun(Gvec_num, param, lambda, nu));
       %% =====================================================
    % Diagnostico geometrico
    %% =====================================================
% 
%   G = full(H_SymFun( ...
%         Gvec_num, param, lambda, nu));
%  
%     G = 0.5*(G + G.');
% 
%     % Calcular autovalores
% autovalores = min(eig(G));
% 
% disp(autovalores);
% 
% autovalores = min(eig(A));
% 
% disp(autovalores);
%     B = full(A_skewFun( ...
%         Gvec_num, param, lambda, nu));
% 
%     B = 0.5*(B - B.');

    %% =====================================================
    % Avaliacao das restricoes
    %% =====================================================

    if has_g

        g_val = full(g_fun(Gvec_num,param));

        Jg = full(JgFun(Gvec_num,param));

    else

        g_val = [];
        Jg = [];

    end

    if has_h

        h_val = full(h_fun(Gvec_num,param));

        Jh = full(JhFun(Gvec_num,param));

    else

        h_val = [];
        Jh = [];

    end

    %% =====================================================
    % Residuo KKT
    %% =====================================================

    r_dual = gradL;

    if has_g
        r_dual = r_dual + Jg'*nu;
    end

    if has_h
        r_dual = r_dual + Jh'*lambda;
    end

    r_prim_g = [];
    r_cent = [];

    if has_g

        r_prim_g = g_val + s;

        r_cent = nu.*s - mu*ones(m,1);

    end

    r_prim_h = [];

    if has_h

        r_prim_h = h_val;

    end

    res_norm = norm([ ...
        r_dual;
        r_prim_g;
        r_prim_h;
        r_cent]);

    %% =====================================================
    % Criterio de convergencia
    %% =====================================================

    if res_norm < tol

        fprintf('Convergencia na iteracao %d\n',k);

        break

    end

    %% =====================================================
    % Sistema KKT
    %% =====================================================

    if has_g && has_h

        S = spdiags(s,0,m,m);

        Nmat = spdiags(nu,0,m,m);

        KKT = [ A,                 Jh',             Jg',             sparse(d*N,m);
                Jh,                sparse(p,p),     sparse(p,m),     sparse(p,m);
                Jg,                sparse(m,p),     sparse(m,m),     speye(m);
                sparse(m,d*N),     sparse(m,p),     S,               Nmat ];

        rhs = -[ r_dual;
                 r_prim_h;
                 r_prim_g;
                 r_cent ];

    elseif has_g && ~has_h

        S = spdiags(s,0,m,m);

        Nmat = spdiags(nu,0,m,m);

        KKT = [ A,                 Jg',             sparse(d*N,m);
                Jg,                sparse(m,m),    speye(m);
                sparse(m,d*N),     S,               Nmat ];

        rhs = -[ r_dual;
                 r_prim_g;
                 r_cent ];

    elseif has_h && ~has_g

        KKT = [ A,                 Jh';
                Jh,                sparse(p,p) ];

        rhs = -[ r_dual;
                 r_prim_h ];

    else

        KKT = A;

        rhs = -r_dual;

    end

    %% =====================================================


    delta = (KKT+reg*eye(size(KKT))) \rhs;

    %% =====================================================
    % Extracao das direcoes
    %% =====================================================

    idx = 1;

    delta_G = delta(idx:idx+d*N-1);

    idx = idx + d*N;

    if has_h

        dlambda = delta(idx:idx+p-1);

        idx = idx + p;

    else

        dlambda = [];

    end

    if has_g

        dnu = delta(idx:idx+m-1);

        idx = idx + m;

        ds = delta(idx:idx+m-1);

    else

        dnu = [];
        ds = [];

    end

    %% =====================================================
    % Limitacao do passo
    %% =====================================================

    alpha = 1;

    % Fraction-to-boundary para s e nu

    tau = 0.995;

    if has_g

        idxs = ds < 0;

        if any(idxs)

            alpha = min(alpha, ...
                tau*min(-s(idxs)./ds(idxs)));

        end

        idxn = dnu < 0;

        if any(idxn)

            alpha = min(alpha, ...
                tau*min(-nu(idxn)./dnu(idxn)));

        end

    end



    %% =====================================================
    % Atualizacao geometrica em SL(n)
    %% =====================================================

    j = 1;

    for i = 1:N
        eta_i = delta_G(j:j+d-1);

        S_i = full(casadi_funs.S_fun(alpha*eta_i));

        Gi_1 = reshape( ...
            Gvec_num((i-1)*n^2+1:i*n^2),n,n);

        Gi = Gi_1*expm(S_i);

        Gvec_num((i-1)*n^2+1:i*n^2) = Gi(:);

        j = j+d;

    end

    %% =====================================================
    % Atualizacao dos multiplicadores
    %% =====================================================

    if has_g

        s = s + alpha*ds;

        nu = nu + alpha*dnu;

        % Atualizacao do parametro de barreira
        mu = 0.5*(s'*nu)/max(1,m);

    end

    if has_h

        lambda = lambda + alpha*dlambda;

    end

    %% =====================================================
    % Informacoes da iteracao
    %% =====================================================

    if order == 1

        fprintf(['iter %d |res| = %.3e |alpha| = %.3f ' ...
                 '|mu| = %.3e\n'], ...
                 k,res_norm,alpha,mu);

    end

end

%% =========================================================
% Saida
%% =========================================================

G_out = cell(N,1);

for i = 1:N

    G_out{i} = reshape( ...
        Gvec_num((i-1)*n^2+1:i*n^2),n,n);

end

end
