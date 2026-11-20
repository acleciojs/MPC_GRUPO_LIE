% COMPARAR_ADMM_PRECONDICIONADO
% Duas execucoes separadas (mesmo problema), uma scalar e outra curvature.
% Mude N e M para testar horizontes maiores. T_sim=1 evita misturar
% trajetorias fechadas diferentes na comparacao da PRIMEIRA QP.
clear functions; clc; close all;
cfg=struct();cfg.N=40;cfg.M=10;cfg.dt=0.01;cfg.T_sim=1;
cfg.max_sqp=1;cfg.rho=20;cfg.max_admm=2500;
cfg.tol_abs=1e-5;cfg.tol_rel=1e-5;
cfg.tol_gap_inf=1e-7;cfg.tol_eq_lin=1e-5;cfg.tol_ineq_lin=1e-7;
cfg.hessian_reg_base=1e-3;cfg.hessian_reg_floor=1e-4;
cfg.qp_solver='qpoases';cfg.plot_results=false;
cfg.verbose_admm=false;cfg.compare_central=false;
cfg.warm_start=true;cfg.warm_state_mode='rollout';
cfg.precond_power=0.5;cfg.precond_clamp=[0.2 5];
modos={'scalar','curvature'};
res=cell(2,1);tempos=zeros(2,1);
for k=1:2
    cfg.precond_mode=modos{k};
    fprintf('=== Testando %s ===\n',modos{k});
    t=tic;res{k}=nmpc_multirotor_usando_sqp_lie_admm(cfg);
    tempos(k)=toc(t);
    fprintf('%s | status=%s | tempo=%.2fs\n', ...
        modos{k},res{k}.status,tempos(k));
end
fprintf('\n  modo           tempo(s)      ADMM iter       gap_inf\n');
for k=1:2
    if res{k}.completed_steps>0
        fprintf('  %-12s %10.2f %12d %13.3e\n', ...
            modos{k},tempos(k),res{k}.admm_iters(1),res{k}.gap(1));
    else
        fprintf('  %-12s %10.2f       falhou (%s)\n', ...
            modos{k},tempos(k),res{k}.status);
    end
end
% Tempos incluem geracao simbolica T_GLOBAL/S_GLOBAL; para isolar apenas
% o ADMM, adicione tic/toc ao redor de resolver_admm_qps no sqp_lie_admm.
