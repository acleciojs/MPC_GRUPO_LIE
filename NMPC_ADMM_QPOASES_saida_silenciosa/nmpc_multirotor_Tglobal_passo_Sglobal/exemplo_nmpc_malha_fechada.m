% EXEMPLO_NMPC_MALHA_FECHADA
% Edite os paths de CasADi e do modelo fisico antes de executar.
clear functions; clc; close all;
cfg=struct();
cfg.N=40; cfg.M=10; cfg.dt=0.01;
cfg.T_sim=20; cfg.max_sqp=1;
cfg.rho=20; cfg.max_admm=2500;
cfg.tol_abs=1e-5;cfg.tol_rel=1e-5;
cfg.tol_gap_inf=1e-7; cfg.tol_eq_lin=1e-5;
cfg.tol_ineq_lin=1e-7;
cfg.warm_start=true;
cfg.warm_state_mode='shift'; % shift dos estados e controles do MPC
cfg.qp_solver='qpoases';
cfg.plot_results=true;
out=nmpc_multirotor_usando_sqp_lie_admm(cfg);
fprintf('Malha fechada: %d/%d amostras | status=%s\n', ...
    out.completed_steps,cfg.T_sim,out.status);
