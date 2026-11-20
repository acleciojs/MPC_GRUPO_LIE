% COMPARAR_ADMM_WARM_START  Comparacao malha fechada no mesmo problema.
% Requer as funcoes do modelo e CasADi, alem de qpOASES.
clear functions;clc;
cfg=struct();
cfg.N=40;cfg.M=10;cfg.dt=0.01;cfg.T_sim=5;
cfg.max_sqp=1;cfg.rho=20;cfg.max_admm=2500;
cfg.precond_mode='curvature';cfg.precond_power=0.5;
cfg.precond_clamp=[0.2 5];
cfg.tol_abs=1e-5;cfg.tol_rel=1e-5;cfg.tol_gap_inf=1e-7;
cfg.tol_eq_lin=1e-5;cfg.tol_ineq_lin=1e-7;
cfg.warm_start=true;cfg.warm_state_mode='rollout';
cfg.qp_solver='qpoases';
cfg.verbose=false;cfg.verbose_admm=false;cfg.plot_results=false;

% Mesma trajetoria inicial, mesma configuracao, apenas warm do ADMM muda.
cfg.admm_warm_start=false;
fprintf('\n===== ADMM SEM WARM =====\n');tic;
a=nmpc_multirotor_usando_sqp_lie_admm(cfg);
tcold=toc;

cfg.admm_warm_start=true;
cfg.admm_warm_dual_damping=0.95;
cfg.admm_warm_primal_max_inf=0.5;
fprintf('\n===== ADMM COM WARM =====\n');tic;
b=nmpc_multirotor_usando_sqp_lie_admm(cfg);
twarm=toc;

k=min(a.completed_steps,b.completed_steps);
if k>=1
   result=table((1:k)',a.admm_iters(1:k)',b.admm_iters(1:k)', ...
       a.gap(1:k)',b.gap(1:k)', b.admm_warm_used(1:k)', ...
       'VariableNames',{'amostra','iters_sem','iters_com',...
       'gap_sem','gap_com','warm_usado'});
   disp(result);
   fprintf('Iteracoes acumuladas: SEM=%d | COM=%d\n', ...
       sum(a.admm_iters(1:k)),sum(b.admm_iters(1:k)));
end
fprintf('Tempos totais MATLAB (incluem derivadas): SEM=%.3fs COM=%.3fs\n',...
    tcold,twarm);
if a.completed_steps~=b.completed_steps
    warning('Execucoes concluíram quantidades diferentes de amostras');
end
