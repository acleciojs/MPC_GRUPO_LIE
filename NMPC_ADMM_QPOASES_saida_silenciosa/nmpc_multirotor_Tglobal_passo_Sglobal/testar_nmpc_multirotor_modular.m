% TESTAR_NMPC_MULTIROTOR_MODULAR
% Coloque esta pasta, CasADi e a pasta ./Modelo do seu projeto no path.
% Necessario: Init.m, Equilibrio.m, IntegrationEuler.m, Attitude_error.m.
clear; clc; close all;

% TESTE PEQUENO: primeiro confirme CasADi, dimensoes e solver local.
cfg=struct();
cfg.N=6;             % depois substitua por 40
cfg.M=3;             % depois substitua por 10
cfg.dt=0.01;
cfg.T_sim=1;         % depois substitua por 100 para malha fechada
cfg.max_sqp=1;
cfg.rho=10;
cfg.max_admm=2500;
cfg.tol_abs=1e-5;
cfg.tol_rel=1e-5;
cfg.hessian_reg_base=1e-3;
cfg.hessian_reg_floor=1e-4;
cfg.qp_solver='qpoases';
cfg.compare_central=true;
cfg.verificar_jacobianas=true; % Conferir Jg/Jh do usuario contra jacobiana direta
cfg.verbose=true;
cfg.verbose_admm=true;
cfg.verbose_every=100;
cfg.p_init=[1;1;1];
cfg.p_ref=[1.2;1.2;1.2];
cfg.plot_results=true;

out=nmpc_multirotor_usando_sqp_lie_admm(cfg);
disp(out.status);
disp(out.derivadas);
if out.completed_steps>0
    fprintf('Erro de posicao apos a primeira amostra: %.6e\n',out.err(end));
    if ~isempty(out.sqp{1}.central_comparison)
        fprintf('Erro relativo ADMM/central: %.6e\n', ...
            out.sqp{1}.central_comparison(end).erro_relativo);
    end
end
