% EXEMPLO_ADMM_PRECONDICIONADO
% Requer Init, Equilibrio, IntegrationEuler, Attitude_error, CasADi, qpOASES.
clear functions; clc; close all;
cfg=struct();
cfg.N=40; cfg.M=10; cfg.dt=0.01; cfg.T_sim=1;
cfg.max_sqp=1;
cfg.rho=20; cfg.max_admm=2500;
cfg.tol_abs=1e-5; cfg.tol_rel=1e-5;
cfg.tol_gap_inf=1e-7; cfg.tol_eq_lin=1e-5; cfg.tol_ineq_lin=1e-7;
cfg.hessian_reg_base=1e-3;cfg.hessian_reg_floor=1e-4;
cfg.qp_solver='qpoases';
cfg.warm_start=true;cfg.warm_state_mode='rollout';
cfg.plot_results=false;cfg.verbose_admm=true;

% Habilita penalizacao DIAGONAL POSITIVA nas interfaces.
cfg.precond_mode='curvature';      % ou 'scalar' (ADMM original)
cfg.precond_power=0.5;           % expoente moderado entre 0 e 1
cfg.precond_clamp=[0.2 5];      % limites dos pesos relativos
cfg.compare_central=false;      % true: compara direcao com QP central

out=nmpc_multirotor_usando_sqp_lie_admm(cfg);
fprintf('Status=%s | amostras=%d\n',out.status,out.completed_steps);
if out.completed_steps>0
    fprintf('ADMM=%d | gap=%g | eq=%g\n', ...
        out.admm_iters(end),out.gap(end),out.eq_lin(end));
    info=out.sqp{end}.sqp(end).admm;
    disp('Pesos relativos da primeira interface:');
    disp(info.metric_weights{1}(:)');
end
