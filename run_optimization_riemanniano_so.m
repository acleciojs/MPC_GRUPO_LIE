function G_out = run_optimization_riemanniano_so(sym_data, casadi_funs, R0, max_iter, tol)

%% =========================================================
%% Extração de dados
%% =========================================================
N = sym_data.N;
n = sym_data.n;
dim = sym_data.dim;
dim_total = sym_data.dim_total;

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
    g_fun = casadi_funs.g_fun;
    JgFun = casadi_funs.JgFun;
else
    g_fun = [];
    JgFun = [];
end

if has_h
    h_fun = casadi_funs.h_fun;
    JhFun = casadi_funs.JhFun;
else
    h_fun = [];
    JhFun = [];
end

%% =========================================================
%% Inicialização primal
%% =========================================================
Rvec_num = zeros(dim_total,1);
for i = 1:N
    Ri = R0{i};
    [U,~,V] = svd(Ri);
    Ri = U*V';
    if det(Ri) < 0
        Ri(:,1) = -Ri(:,1);
    end
    Rvec_num((i-1)*dim+1:i*dim) = Ri(:);
end

%% =========================================================
%% Inicialização dual - seguindo RIPM (Seção 3.3)
%% =========================================================
if has_g
    g_val = full(g_fun(Rvec_num));
    s  = max(1, -g_val + 1);  % slack variables S = -g(X) + 1
    nu = ones(m,1);            % multiplicadores Z
    mu = (nu'*s)/m;            % parâmetro de barreira inicial
else
    s = []; nu = []; mu = [];
end

if has_h
    lambda = zeros(p,1);
else
    lambda = [];
end

%% Parâmetros RIPM (baseado no Teorema 5.3 e Seção 6)
theta = 0.5;           % fator de backtracking (Step 3b)
beta = 1e-4;           % parâmetro de Armijo (Seção 8.1)
gamma_k = 0.9;         % γ_k inicial para centralidade (Step 3a)
reg_hessian = 1e-8;    % regularização para Hessiana

% Inicialização para τ�? e τ₂ (Eq. 30) - SOMENTE se has_g = true
if has_g
    ZdotS = nu .* s;
    tau1 = min(ZdotS) / (nu'*s / m);
    % Chamada condicional para compute_KKT_norm
    if has_h
        F0_norm = compute_KKT_norm(Rvec_num, gradFun, ...
                                   has_g, g_fun, JgFun, s, nu, ...
                                   has_h, h_fun, JhFun, lambda);
    else
        F0_norm = compute_KKT_norm(Rvec_num, gradFun, ...
                                   has_g, g_fun, JgFun, s, nu, ...
                                   false, [], [], []);
    end
    tau2 = (nu'*s) / max(F0_norm, eps);
else
    tau1 = 1; tau2 = 1;
end

% Parâmetros para atualização de μ e γ (Teorema 5.3)
gamma_hat = 0.5;  % limite inferior para γ_k

% Para caso sem restrições, usar Newton padrão
if ~has_g && ~has_h
    fprintf('Modo: Sem restrições - usando Newton padrão\n');
end

%% =========================================================
%% Funções auxiliares
%% =========================================================
skew = @(M) (M-M')/2;
proj_tan = @(R,M) R*skew(R'*M);

%% =========================================================
%% LOOP PRINCIPAL
%% =========================================================
for k = 1:max_iter

    %% Avaliações das funções no ponto atual
    gradL = full(gradFun(Rvec_num));
    H = full(HFun(Rvec_num));
   % Garantir simetria

    if has_g
        g_val = full(g_fun(Rvec_num));
        Jg = full(JgFun(Rvec_num));
        % Ajuste de dimensões
        if size(Jg,1) ~= m
            Jg = Jg';
        end
        % Verificar dimensões
        if size(Jg,2) ~= dim_total
            error('Dimensão incorreta de Jg: esperado %d colunas, tem %d', dim_total, size(Jg,2));
        end
    else
        g_val = [];
        Jg = [];
    end
    
    if has_h
        h_val = full(h_fun(Rvec_num));
        Jh = full(JhFun(Rvec_num));
        if size(Jh,1) ~= p
            Jh = Jh';
        end
        if size(Jh,2) ~= dim_total
            error('Dimensão incorreta de Jh: esperado %d colunas, tem %d', dim_total, size(Jh,2));
        end
    else
        h_val = [];
        Jh = [];
    end

    %% Resíduos KKT conforme Eq. (5)
    % grad_X L(X,Y,Z) - gradiente do Lagrangiano
    r_dual = gradL;
    if has_g && ~isempty(nu)
        r_dual = r_dual + Jg' * nu;
    end
    if has_h && ~isempty(lambda)
        r_dual = r_dual + Jh' * lambda;
    end

    % Resíduos das restrições
    if has_h
        r_prim_h = h_val;  % h(X)
    else
        r_prim_h = [];
    end

    if has_g
        r_prim_g = g_val + s;  % g(X) + S
        r_cent   = nu .* s;     % Z ⊙ S (complementaridade)
    else
        r_prim_g = [];
        r_cent = [];
    end

    % Norma do resíduo F(W) - Eq. (5)
    % Montagem cuidadosa para evitar problemas com células vazias
    F_vec = r_dual(:);
    if ~isempty(r_prim_h)
        F_vec = [F_vec; r_prim_h(:)];
    end
    if ~isempty(r_prim_g)
        F_vec = [F_vec; r_prim_g(:)];
    end
    if ~isempty(r_cent)
        F_vec = [F_vec; r_cent(:)];
    end
    res_norm = norm(F_vec);
    
    % Verificar convergência
    if res_norm < tol
        fprintf('Convergência alcançada na iteração %d (res_norm = %.2e)\n', k, res_norm);
        break
    end

    %% Escolha de σ_k e �?_k/μ_k conforme Seção 6 e Eq. (32)
    sigma_k = min(0.5, sqrt(res_norm));  % σ_k ∈ (0,1), Eq. (32) e Seção 8.1
    
    if has_g
        % �?_k ∈ [⟨Z,S⟩/m, ||F(W)||/√m] (Seção 6)
        rho_k = (nu'*s) / m;  % Valor mínimo (usado na Seção 8.1)
        mu_k = sigma_k * rho_k;  % Termo de perturbação
    else
        mu_k = 0;
        rho_k = 0;
    end

    %% =====================================================
    %% Sistema KKT - Forma perturbada Eq. (18)
    %% =====================================================
    if has_g
        S_diag = diag(s);
        N_diag = diag(nu);
    end

    % Montagem do sistema linear conforme Eq. (18) - VERSÃO CORRIGIDA
    if has_g && has_h
        % Matriz completa com igualdade e desigualdade
        % Estrutura: [H    Jh'   Jg'   0;
        %             Jh   0     0     0;
        %             Jg   0     0     I;
        %             0    0     S     Z]
        n_total = dim_total + p + 2*m;
        KKT = zeros(n_total, n_total);
        
        % Preencher blocos
        KKT(1:dim_total, 1:dim_total) = H;
        KKT(1:dim_total, dim_total+1:dim_total+p) = Jh';
        KKT(1:dim_total, dim_total+p+1:dim_total+p+m) = Jg';
        
        KKT(dim_total+1:dim_total+p, 1:dim_total) = Jh;
        
        KKT(dim_total+p+1:dim_total+p+m, 1:dim_total) = Jg;
        KKT(dim_total+p+1:dim_total+p+m, dim_total+p+m+1:end) = eye(m);
        
        KKT(dim_total+p+m+1:end, dim_total+p+1:dim_total+p+m) = S_diag;
        KKT(dim_total+p+m+1:end, dim_total+p+m+1:end) = N_diag;
        
        % RHS: -F(w) + μ_k ê, onde ê = (0_x,0,0,e)
        rhs = zeros(n_total, 1);
        rhs(1:dim_total) = -r_dual(:);
        rhs(dim_total+1:dim_total+p) = -r_prim_h(:);
        rhs(dim_total+p+1:dim_total+p+m) = -r_prim_g(:);
        rhs(dim_total+p+m+1:end) = -r_cent(:) + mu_k;

    elseif has_g && ~has_h
        % Apenas desigualdades
        % Estrutura: [H    Jg'   0;
        %             Jg   0     I;
        %             0    S     Z]
        n_total = dim_total + 2*m;
        KKT = zeros(n_total, n_total);
        
        KKT(1:dim_total, 1:dim_total) = H;
        KKT(1:dim_total, dim_total+1:dim_total+m) = Jg';
        
        KKT(dim_total+1:dim_total+m, 1:dim_total) = Jg;
        KKT(dim_total+1:dim_total+m, dim_total+m+1:end) = eye(m);
        
        KKT(dim_total+m+1:end, dim_total+1:dim_total+m) = S_diag;
        KKT(dim_total+m+1:end, dim_total+m+1:end) = N_diag;
        
        rhs = zeros(n_total, 1);
        rhs(1:dim_total) = -r_dual(:);
        rhs(dim_total+1:dim_total+m) = -r_prim_g(:);
        rhs(dim_total+m+1:end) = -r_cent(:) + mu_k;

    elseif has_h && ~has_g
        % Apenas igualdades
        % Estrutura: [H    Jh';
        %             Jh   0]
        n_total = dim_total + p;
        KKT = zeros(n_total, n_total);
        
        KKT(1:dim_total, 1:dim_total) = H;
        KKT(1:dim_total, dim_total+1:end) = Jh';
        KKT(dim_total+1:end, 1:dim_total) = Jh;
        
        rhs = zeros(n_total, 1);
        rhs(1:dim_total) = -r_dual(:);
        rhs(dim_total+1:end) = -r_prim_h(:);

    else
        % Sem restrições - Newton padrão
        KKT = H;
        rhs = -r_dual(:);
    end

    % Regularização para garantir não-singularidade (Teorema 3.1)
    KKT = KKT + reg_hessian * eye(size(KKT));
    
    % Resolução do sistema linear
    try
        delta = KKT \ rhs;
    catch ME
        warning('Sistema KKT singular na iteração %d. Usando regularização maior. Erro: %s', k, ME.message);
        KKT = KKT + 1e-6 * eye(size(KKT));
        delta = KKT \ rhs;
    end

    %% Extração das direções ΔW = (ΔX, ΔY, ΔZ, ΔS)
    idx = 1;
    deta = delta(idx:idx+dim_total-1);  % ΔX
    idx = idx + dim_total;

    if has_h
        dlambda = delta(idx:idx+p-1);    % ΔY (multiplicadores igualdade)
        idx = idx + p;
    else
        dlambda = [];
    end

    if has_g
        dnu = delta(idx:idx+m-1);        % ΔZ (multiplicadores desigualdade)
        idx = idx + m;
        ds  = delta(idx:end);            % ΔS
    else
        dnu = [];
        ds = [];
    end

    %% =====================================================
    %% Line search - adaptado para cada caso
    %% =====================================================
    alpha = 1.0;

    % Apenas para casos com desigualdades (has_g = true)
    if has_g && ~isempty(ds) && ~isempty(dnu)
        % Garantir que (z,s) > 0 após atualização (Eq. 17)
        idxs = ds < 0;
        if any(idxs)
            alpha_s = min(-s(idxs)./ds(idxs));
            alpha = min(alpha, 0.9995 * alpha_s);  % Fator de segurança
        end

        idxn = dnu < 0;
        if any(idxn)
            alpha_nu = min(-nu(idxn)./dnu(idxn));
            alpha = min(alpha, 0.9995 * alpha_nu);
        end
        
        % Verificar condições de centralidade de forma simplificada
        s_test = s + alpha * ds;
        nu_test = nu + alpha * dnu;
        
        % Garantir positividade
        if any(s_test <= 0) || any(nu_test <= 0)
            alpha = 0.5 * alpha;
        end
        
        % Condição f^I(α) ≥ 0 (simplificada) - apenas se tem desigualdades
        if has_g
            ZdotS_test = nu_test .* s_test;
            if min(ZdotS_test) < gamma_k * tau1 * (nu_test'*s_test)/m
                alpha = 0.5 * alpha;
            end
        end
    end

    %% Condição de Armijo (Step 3b) - Eq. (33) - adaptada
    phi_W = res_norm^2;  % φ(W) = ||F(W)||²
    
    % Calcular ⟨grad φ, ΔW⟩ (Lema 6.1) - adaptado para cada caso
    if has_g
        gradphi_dot_delta = 2 * (-phi_W + sigma_k * rho_k * (nu'*s));
    elseif has_h
        % Para apenas igualdades, o gradiente da função mérito é diferente
        gradphi_dot_delta = -2 * phi_W;  % Aproximação
    else
        % Sem restrições - direção de Newton padrão
        gradphi_dot_delta = -2 * phi_W;
    end
    
    % Verificar se é direção de descida
    if gradphi_dot_delta >= 0
        if res_norm > 1e-6
            warning('Direção não é de descida na iteração %d. gradphi_dot_delta = %.2e', k, gradphi_dot_delta);
        end
        gradphi_dot_delta = -eps;
    end
    
    % Backtracking line search
    alpha_back = alpha;
    t = 0;
    max_backtrack = 20;
    
    while t < max_backtrack
        % Tentativa de passo primal
        Rvec_trial = update_primal(Rvec_num, deta, alpha_back, N, dim, n, proj_tan);
        
        % Atualização dual para teste
        if has_g
            s_trial = s + alpha_back * ds;
            nu_trial = nu + alpha_back * dnu;
            % Garantir positividade numérica
            s_trial = max(s_trial, 1e-12);
            nu_trial = max(nu_trial, 1e-12);
        else
            s_trial = [];
            nu_trial = [];
        end
        
        if has_h
            lambda_trial = lambda + alpha_back * dlambda;
        else
            lambda_trial = [];
        end
        
        % Calcular φ(W(α)) - chamada condicional
        if has_g && has_h
            phi_trial = compute_phi(Rvec_trial, gradFun, ...
                                    true, g_fun, JgFun, s_trial, nu_trial, ...
                                    true, h_fun, JhFun, lambda_trial);
        elseif has_g && ~has_h
            phi_trial = compute_phi(Rvec_trial, gradFun, ...
                                    true, g_fun, JgFun, s_trial, nu_trial, ...
                                    false, [], [], []);
        elseif ~has_g && has_h
            phi_trial = compute_phi(Rvec_trial, gradFun, ...
                                    false, [], [], [], [], ...
                                    true, h_fun, JhFun, lambda_trial);
        else
            phi_trial = compute_phi(Rvec_trial, gradFun, ...
                                    false, [], [], [], [], ...
                                    false, [], [], []);
        end
        
        % Condição de Armijo (Eq. 33)
        if phi_trial - phi_W <= alpha_back * beta * gradphi_dot_delta
            break;
        end
        
        alpha_back = theta * alpha_back;
        t = t + 1;
        
        if alpha_back < 1e-12
            if res_norm > 1e-6
                warning('Passo muito pequeno na iteração %d', k);
            end
            break;
        end
    end
    
    alpha = alpha_back;

    %% Atualização primal (Eq. 6)
    Rvec_num = update_primal(Rvec_num, deta, alpha, N, dim, n, proj_tan);

    %% Atualização dual (Eq. 6) - apenas se aplicável
    if has_g && ~isempty(ds) && ~isempty(dnu)
        s  = s  + alpha * ds;
        nu = nu + alpha * dnu;
        s  = max(s, 1e-12);
        nu = max(nu, 1e-12);
        
        mu = (nu' * s) / m;
        
        % Atualizar τ�? e τ₂ para próxima iteração (Eq. 30)
        ZdotS = nu .* s;
        tau1 = min(ZdotS) / (nu'*s / m + eps);
        
        % Chamada condicional para compute_KKT_norm
        if has_h
            F_norm = compute_KKT_norm(Rvec_num, gradFun, ...
                                      true, g_fun, JgFun, s, nu, ...
                                      true, h_fun, JhFun, lambda);
        else
            F_norm = compute_KKT_norm(Rvec_num, gradFun, ...
                                      true, g_fun, JgFun, s, nu, ...
                                      false, [], [], []);
        end
        tau2 = (nu'*s) / max(F_norm, eps);
    end

    if has_h && ~isempty(dlambda)
        lambda = lambda + alpha * dlambda;
    end

    %% Atualizar parâmetros (Teorema 5.3)
    % γ_k decrescente (Step 3a) - apenas se tem desigualdades
    if has_g
        gamma_k = max(gamma_hat, gamma_k * 0.95);
    end
    
    % Para convergência quadrática (Teorema 5.3(2))
    if res_norm < 1e-2
        sigma_k = min(0.5, res_norm);
    end

    %% Output diagnóstico - adaptado
%     if mod(k,10)==1 || k==1 || res_norm < 1e-4
%         if has_g && has_h
%             fprintf('iter %4d | ||F||=%.2e | α=%.2e | μ=%.2e | ⟨z,s⟩=%.2e | min(zs)=%.2e | κ=%.2e\n',...
%                     k, res_norm, alpha, mu, nu'*s, min(nu.*s), cond(KKT));
%         elseif has_g
%             fprintf('iter %4d | ||F||=%.2e | α=%.2e | μ=%.2e | ⟨z,s⟩=%.2e | min(zs)=%.2e | κ=%.2e\n',...
%                     k, res_norm, alpha, mu, nu'*s, min(nu.*s), cond(KKT));
%         elseif has_h
%             fprintf('iter %4d | ||F||=%.2e | α=%.2e | ||h||=%.2e | κ=%.2e\n',...
%                     k, res_norm, alpha, norm(r_prim_h), cond(KKT));
%         else
%             fprintf('iter %4d | ||∇f||=%.2e | α=%.2e | κ(H)=%.2e\n',...
%                     k, res_norm, alpha, cond(H));
%         end
    end
end

%% Preparar saída
G_out = cell(N,1);
for i = 1:N
    Ri = reshape(Rvec_num((i-1)*dim+1:i*dim), n, n);
    [U,~,V] = svd(Ri);
    Ri = U*V';
    if det(Ri) < 0
        Ri(:,1) = -Ri(:,1);
    end
    G_out{i} = Ri;
end

fprintf('Otimização concluída após %d iterações (res_final = %.2e)\n', k, res_norm);
end

%% =========================================================
%% Funções auxiliares
%% =========================================================

function Rvec_new = update_primal(Rvec, deta, alpha, N, dim, n, proj_tan)
    % Atualização primal via retração (Eq. 6)
    Rvec_new = Rvec;
    j = 1;
    for i = 1:N
         alpha = 1.0;
        eta_i = alpha * deta(j:j+dim-1);
        
        Ri = reshape(Rvec((i-1)*dim+1:i*dim), n, n);
        eta_i = reshape(eta_i, n, n);
        
        % Projeção no espaço tangente (para variedade Stiefel/SO(n))
        eta_proj = proj_tan(Ri, eta_i);
        
        % Atualização via retração exponencial
        % Para SO(n): R_X(ξ) = X * expm(X^T ξ)
        Xi = Ri' * eta_proj;
        % Garantir que Xi é anti-simétrica
        Xi = (Xi - Xi')/2;
        
        % Usar expm para matrizes pequenas/médias
        try
            Ri_new = Ri * expm(Xi);
        catch
            % Fallback para matrizes maiores - aproximação de Cayley
            I = eye(n);
            Ri_new = Ri * (I + Xi/2) / (I - Xi/2);
        end
        
        % Projeção de volta à variedade SO(n) (garantir ortogonalidade)
        [U,~,V] = svd(Ri_new);
        Ri_new = U * V';
        
        % Garantir determinante positivo (SO(n) tem det = +1)
        if det(Ri_new) < 0
            Ri_new(:,1) = -Ri_new(:,1);
        end
        
        Rvec_new((i-1)*dim+1:i*dim) = Ri_new(:);
        j = j + dim;
    end
end

function phi = compute_phi(Rvec, gradFun, ...
                           has_g, g_fun, JgFun, s, nu, ...
                           has_h, h_fun, JhFun, lambda)
    % Calcular ||F(W)||² conforme Eq. (5)
    
    gradL = full(gradFun(Rvec));
    
    if has_g && ~isempty(nu)
        g_val = full(g_fun(Rvec));
        Jg = full(JgFun(Rvec));
        if size(Jg,1) ~= length(nu)
            Jg = Jg';
        end
    end
    
    if has_h && ~isempty(lambda)
        h_val = full(h_fun(Rvec));
        Jh = full(JhFun(Rvec));
        if size(Jh,1) ~= length(lambda)
            Jh = Jh';
        end
    end
    
    % r_dual = grad_x L
    r_dual = gradL;
    if has_g && ~isempty(nu)
        r_dual = r_dual + Jg' * nu;
    end
    if has_h && ~isempty(lambda)
        r_dual = r_dual + Jh' * lambda;
    end
    
    % Montagem do vetor F(w) - tratamento cuidadoso
    F_vec = r_dual(:);
    
    if has_h && ~isempty(lambda)
        F_vec = [F_vec; h_val(:)];
    end
    
    if has_g && ~isempty(s) && ~isempty(nu)
        F_vec = [F_vec; g_val(:) + s(:)];
        F_vec = [F_vec; nu(:) .* s(:)];
    end
    
    phi = norm(F_vec)^2;
end

function F_norm = compute_KKT_norm(Rvec, gradFun, ...
                                   has_g, g_fun, JgFun, s, nu, ...
                                   has_h, h_fun, JhFun, lambda)
    % Calcular ||F(W)|| (norma do resíduo KKT)
    % Similar a compute_phi mas retorna a norma, não a norma ao quadrado
    
    gradL = full(gradFun(Rvec));
    
    if has_g && ~isempty(nu)
        g_val = full(g_fun(Rvec));
        Jg = full(JgFun(Rvec));
        if size(Jg,1) ~= length(nu)
            Jg = Jg';
        end
    end
    
    if has_h && ~isempty(lambda)
        h_val = full(h_fun(Rvec));
        Jh = full(JhFun(Rvec));
        if size(Jh,1) ~= length(lambda)
            Jh = Jh';
        end
    end
    
    % r_dual
    r_dual = gradL;
    if has_g && ~isempty(nu)
        r_dual = r_dual + Jg' * nu;
    end
    if has_h && ~isempty(lambda)
        r_dual = r_dual + Jh' * lambda;
    end
    
    % Montagem
    F_vec = r_dual(:);
    
    if has_h && ~isempty(lambda)
        F_vec = [F_vec; h_val(:)];
    end
    
    if has_g && ~isempty(s) && ~isempty(nu)
        F_vec = [F_vec; g_val(:) + s(:)];
        F_vec = [F_vec; nu(:) .* s(:)];
    end
    
    F_norm = norm(F_vec);
end