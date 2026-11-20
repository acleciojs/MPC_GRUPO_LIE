% Exemplo de execucao de SQP geometrica + ADMM precondicionado e warm start.
% Necessario manter Init, Equilibrio, IntegrationEuler, Attitude_error no path.
clc;clear functions;close all;
cfg=struct();
cfg.N=40;cfg.M=10;cfg.dt=0.01;cfg.T_sim=20;
cfg.max_sqp=1;
cfg.rho=20;cfg.max_admm=2500;
cfg.precond_mode='curvature';
cfg.precond_power=0.5;cfg.precond_clamp=[0.2 5];
cfg.tol_abs=1e-5;cfg.tol_rel=1e-5;
cfg.tol_gap_inf=1e-7;cfg.tol_eq_lin=1e-5;cfg.tol_ineq_lin=1e-7;
cfg.hessian_reg_base=1e-3;cfg.hessian_reg_floor=1e-4;

% Warm start da trajetoria MPC (a planta continua nao linear em closed-loop)
cfg.warm_start=true;cfg.warm_state_mode='rollout';

% NOVO: warm start do ADMM entre as amostras e SQPs
cfg.admm_warm_start=true;
cfg.admm_warm_dual_damping=0.95;
cfg.admm_warm_primal_damping=1.0;
cfg.admm_warm_primal_max_inf=0.5;
cfg.admm_warm_local_qp=true;

% interior-point-convex ignora x0 local, mas usa o warm de consenso/duais.
% Use 'qpoases' ou 'qpoases' para testar x0 nas QPs locais.
cfg.qp_solver='qpoases';
cfg.verbose=true;cfg.verbose_admm=true;cfg.verbose_every=100;
cfg.plot_results=true;

out=nmpc_multirotor_usando_sqp_lie_admm(cfg);
if out.completed_steps>0
    disp(table((1:out.completed_steps)',out.admm_iters(:), ...
        out.gap(:),out.admm_warm_used(:), ...
        'VariableNames',{'amostra','iter_ADMM','gap_inf','warm_ADMM'}));
end
