% Exemplo para N=200, M=50 SEM quadprog.
% Depende de CasADi (plugin qpOASES) e modelo do VANT no path:
% Init, Equilibrio, IntegrationEuler, Attitude_error.
clear functions; clc; close all;
cfg=struct();
cfg.N=200;cfg.M=50;cfg.dt=0.01;cfg.T_sim=10;
cfg.max_sqp=1;cfg.rho=20;cfg.max_admm=2500;
cfg.precond_mode='curvature';
cfg.precond_power=0.5;cfg.precond_clamp=[0.2 5];
cfg.tol_abs=1e-5;cfg.tol_rel=1e-5;cfg.tol_gap_inf=1e-7;
cfg.tol_eq_lin=1e-5;cfg.tol_ineq_lin=1e-5;
cfg.hessian_reg_base=1e-3;cfg.hessian_reg_floor=1e-4;
cfg.warm_start=true;cfg.warm_state_mode='rollout';
cfg.admm_warm_start=true;cfg.admm_warm_dual_damping=0.90;
cfg.admm_warm_primal_damping=1.0;
cfg.admm_warm_primal_max_inf=0.5;
cfg.admm_warm_local_qp=true;
cfg.qp_solver='qpoases';
cfg.qp_max_iterations=2000;     % qpOASES nWSR, nao quadprog
cfg.qp_restart_on_failure=true;
cfg.qp_local_eq_tol=1e-6;
cfg.qp_local_ineq_tol=1e-5;
cfg.compare_central=false;      % N=200 QP central muito grande
cfg.verbose=true;cfg.verbose_admm=true;
cfg.verbose_every=100;cfg.plot_results=true;
out=nmpc_multirotor_usando_sqp_lie_admm(cfg);
fprintf('MPC: %s | amostras=%d\n',out.status,out.completed_steps);
if isfield(out,'failed_sqp') && ~isempty(out.failed_sqp)
    disp(out.failed_sqp);
end
