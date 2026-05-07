clc; clear; close all;
import casadi.*
addpath('C:\Users\aclec\OneDrive\Documentos\casadi-3.7.1-windows64-matlab2018b')

%% =========================================================
%% CONFIGURAÇÕES
%% =========================================================
n = 3;
N = 2;
Nrep = 1000;
max_iter = 500;
tol = 1e-5;

fprintf('\n====================================================\n');
fprintf('COMPARAÇÃO EM SO(%d) COM DISTÂNCIA GEODÉSICA\n', n);
fprintf('====================================================\n\n');

%% =========================================================
%% VARI�?VEIS SIMBÓLICAS
%% =========================================================
R_sym = cell(1,N);
for k = 1:N
    R_sym{k} = SX.sym(sprintf('R%d',k), n, n);
end

%% =========================================================
%% MATRIZ ALVO EM SO(n)
%% =========================================================
Rd = cell(1,N);
for k = 1:N
    A = randn(n,n);
    [U,~,V] = svd(A);
    Rk = U*V';
    if det(Rk)<0, Rk(:,1)=-Rk(:,1); end
    Rd{k} = Rk;
end

%% =========================================================
%% FUNÇÃO OBJETIVO GEODÉSICA
%% =========================================================
f = 0;
for k = 1:N
    
    X = Rd{k}' * R_sym{k};
    Omega = logm_casadi(X,25,3);
    
    % garantir elemento em so(n)
    Omega = Omega - (trace(Omega)/n)*eye(n);
    
    f = f + 0.5 * trace(Omega.'*Omega);
end

%% =========================================================
%% RESTRIÇÕES g ≤ 0
%% =========================================================
g = [];
for k = 1:N
    g = [g;
         R_sym{k}(1,1) - 0.5;
         R_sym{k}(2,2) - 0.3];
end
h = [];

%% =========================================================
%% CRIAR PARTES SIMBÓLICAS
%% =========================================================
[sym_euc, casadi_euc]   = create_symbolic_part(f, R_sym, g, h);
%[sym_riem, casadi_riem] = create_symbolic_part_riemanniana_son(f, R_sym, g, h);

%% =========================================================
%% ARMAZENAMENTO
%% =========================================================
f_euc = zeros(Nrep,1);
f_riem = zeros(Nrep,1);
time_euc = zeros(Nrep,1);
time_riem = zeros(Nrep,1);

viol_euc  = zeros(Nrep,1);
viol_riem = zeros(Nrep,1);
feas_euc  = zeros(Nrep,1);
feas_riem = zeros(Nrep,1);

%% =========================================================
%% LOOP PRINCIPAL
%% =========================================================
for rep = 1:Nrep
    
    fprintf('Execução %d/%d\n', rep, Nrep);
    
    %% =====================================================
    %% GERAR PONTO INICIAL VI�?VEL EM SO(n)
    %% =====================================================
    is_feasible = false;
    tentativas = 0;
    
    while ~is_feasible
        
        tentativas = tentativas + 1;
        R0 = cell(1,N);
        
        for k = 1:N
            Rk = randn(n,n);
            [U,~,V] = svd(Rk);
            Rk = U*V';
            if det(Rk)<0, Rk(:,1)=-Rk(:,1); end
            
            % Perturbação
            S = randn(n); 
            S = (S-S')/2;
            S = S/norm(S,'fro')*1.5;
            Rk = Rk*expm(S);
            
            [U,~,V] = svd(Rk);
            Rk = U*V';
            if det(Rk)<0, Rk(:,1)=-Rk(:,1); end
            
            R0{k} = Rk;
        end
        
        % Empilhamento correto (32x1)
        Rvec0 = [];
        for k = 1:N
            Rvec0 = [Rvec0; R0{k}(:)];
        end
        
        g0 = full(casadi_euc.g_fun(Rvec0))
        viol0 = max(g0);
        is_feasible = viol0 <= 1e-8;
        
    end
    
    fprintf('  Ponto inicial viável após %d tentativa(s)\n', tentativas);
    fprintf('  Violação inicial máxima: %.3e\n', viol0);
    
    %% =====================================================
    %% 1�?⃣ EUCLIDIANO
    %% =====================================================
    tic;
    R_euc = run_optimization_sqp(sym_euc, casadi_euc, R0, max_iter, tol);
    time_euc(rep) = toc;
    
    Rvec_euc = [];
    for k = 1:N
        Rvec_euc = [Rvec_euc; R_euc{k}(:)];
    end
    
    f_euc(rep) = full(casadi_euc.f_fun(Rvec_euc));
    g_euc = full(casadi_euc.g_fun(Rvec_euc))
    
    viol_euc(rep) = max(g_euc);
    feas_euc(rep) = viol_euc(rep) <= 1e-12
    
   
    
end

%% =========================================================
%% RESULTADOS
%% =========================================================
fprintf('\n====================================================\n');
fprintf('RESULTADOS\n');
fprintf('====================================================\n');

fprintf('\n--- CUSTO ---\n');
fprintf('Euclidiano:  %.4e ± %.4e\n', mean(f_euc), std(f_euc));
fprintf('Riemanniano: %.4e ± %.4e\n', mean(f_riem), std(f_riem));

fprintf('\n--- TEMPO ---\n');
fprintf('Euclidiano:  %.4e s\n', mean(time_euc));
fprintf('Riemanniano: %.4e s\n', mean(time_riem));

fprintf('\n--- RESTRIÇÕES ---\n');

fprintf('\nEuclidiano:\n');
fprintf('  Violação média positiva: %.2e\n', mean(max(viol_euc,0)));
fprintf('  Máxima violação: %.2e\n', max(viol_euc));
fprintf('  Percentual viável: %.1f%%\n', 100*mean(feas_euc));

fprintf('\nRiemanniano:\n');
fprintf('  Violação média positiva: %.2e\n', mean(max(viol_riem,0)));
fprintf('  Máxima violação: %.2e\n', max(viol_riem));
fprintf('  Percentual viável: %.1f%%\n\n', 100*mean(feas_riem));

%% =========================================================
%% GR�?FICOS
%% =========================================================
figure;
subplot(1,3,1);
boxplot([f_euc,f_riem],'Labels',{'Euclidiano','Riemanniano'});
title('Custo Final');

subplot(1,3,2);
boxplot([time_euc,time_riem],'Labels',{'Euclidiano','Riemanniano'});
title('Tempo');

subplot(1,3,3);
boxplot([max(viol_euc,0),max(viol_riem,0)], ...
        'Labels',{'Euclidiano','Riemanniano'});
title('Violação das Restrições');