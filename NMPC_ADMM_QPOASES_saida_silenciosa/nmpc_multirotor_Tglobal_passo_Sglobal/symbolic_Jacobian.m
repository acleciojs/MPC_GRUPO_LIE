function J = symbolic_Jacobian(f, R_list, S_sym, omega_sym)
% Jacobiana geometrica do pullback avaliada em xi=0.
% Usa o mesmo metodo vetorizado fornecido pelo usuario.
    import casadi.*
    N = numel(R_list);
    n = size(R_list{1},1);
    d = length(omega_sym);
    total_vars = d*N;
    if isa(f,'casadi.SX')
        xi = SX.sym('xi',total_vars,1);
        I_n = SX.eye(n);
    else
        xi = MX.sym('xi',total_vars,1);
        I_n = MX.eye(n);
    end
    S_cell = arrayfun(@(r) substitute(S_sym,omega_sym, ...
        xi((r-1)*d+(1:d))),1:N,'UniformOutput',false);
    S_big = diagcat(S_cell{:});
    R_big = diagcat(R_list{:});
    R_pert_big = R_big * expm_casadi(S_big);
    V = repmat(I_n,N,1);
    R_pert_all = R_pert_big*V;
    R_orig_all = vertcat(R_list{:});
    f_pert = substitute(f,R_orig_all,R_pert_all);
    J_expr = jacobian(f_pert,xi);
    J = substitute(J_expr,xi,zeros(total_vars,1));
end
