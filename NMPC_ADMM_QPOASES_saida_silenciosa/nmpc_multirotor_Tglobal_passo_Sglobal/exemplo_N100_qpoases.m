% Exemplo: tolerancias numericas MAIS flexiveis sem remover restricoes.
clear functions; clc;
cfg = struct();
cfg.N = 100;
cfg.M = 50;
cfg.dt = 0.01;
cfg.T_sim = 10;
cfg.max_sqp = 1;
cfg.rho = 20;
cfg.max_admm = 2500;
cfg.precond_mode = 'curvature';
cfg.precond_power = 0.5;
cfg.precond_clamp = [0.2 5];
cfg.admm_warm_start = true;
cfg.warm_start = true;
cfg.warm_state_mode = 'rollout';
cfg.qp_solver = 'qpoases';
% Aceita erro local semelhante ao medido: ineq=6.497e-6.
cfg.qp_constraint_tolerance = 1e-5;
cfg.qp_local_ineq_tol = 1e-5;
cfg.qp_local_eq_tol = 1e-6;
cfg.qp_optimality_tolerance = 1e-8;
cfg.qp_max_iterations = 800;
% A reconstrução global tem propria tolerancia.
% Se limites fisicos exigirem tolerancia inferior, NAO aumente aqui;
% nesse caso é necessario um refinamento/projecao da QP local.
cfg.tol_ineq_lin = 1e-5;
cfg.tol_eq_lin = 1e-5;
cfg.tol_gap_inf = 1e-7;
cfg.verbose = true;
cfg.verbose_admm = true;
cfg.plot_results = true;
out = nmpc_multirotor_usando_sqp_lie_admm(cfg);
