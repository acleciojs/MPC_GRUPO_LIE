function [H_sym] = symbolic_Hessian_simetrica( ...
    f, G_list, S_sym, omega_sym, tip)
% Hessiana geometrica simetrica: pullback exponencial avaliado em xi=0.
% Usa expm_casadi apenas para derivadas ate segunda ordem no zero.
    import casadi.*
    if nargin < 5 || isempty(tip)
        tip = 0;
    end
    if ~ismember(tip, [0 1 2])
        error('tip deve ser 0, 1 ou 2.');
    end
    N = numel(G_list);
    m = numel(omega_sym);
    total_vars = m * N;
    isSX = isa(f, 'casadi.SX');
    if isSX
        xi = SX.sym('xi', total_vars, 1);
    else
        xi = MX.sym('xi', total_vars, 1);
    end
    R_pert_cell = cell(N, 1);
    for r = 1:N
        idx = (r-1)*m + (1:m);
        xi_r = xi(idx);
        S_r = substitute(S_sym, omega_sym, xi_r);
        R_pert_cell{r} = G_list{r} * expm_casadi(S_r);
    end
    f_pert_cell = substitute({f}, G_list, R_pert_cell);
    f_pert = f_pert_cell{1};
    H_expr = hessian(f_pert, xi);
    H_sym = substitute(H_expr, xi, zeros(total_vars, 1));
end
