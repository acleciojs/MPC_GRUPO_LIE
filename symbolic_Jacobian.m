function J = symbolic_Jacobian(f, R_list, S_sym, omega_sym)
    % Jacobiana simbólica TOTALMENTE vetorizada (sem loops explícitos)
    import casadi.*
    
    N = numel(R_list);
    M = numel(f);
    n = size(R_list{1},1);
    d = length(omega_sym); % Dimensão da álgebra de Lie (extraída de omega_sym)
    total_vars = d * N;
    
    % 1. Vetor global de perturbações
    if isa(f, 'casadi.SX')
        xi = SX.sym('xi', total_vars, 1);
        I_n = SX.eye(n);
    else
        xi = MX.sym('xi', total_vars, 1);
        I_n = MX.eye(n);
    end
    
    % 2. Cria as matrizes de perturbação S_r via arrayfun (sem for)
    S_cell = arrayfun(@(r) substitute(S_sym, omega_sym, xi((r-1)*d + 1 : r*d)), 1:N, 'UniformOutput', false);
    S_big = diagcat(S_cell{:}); % Matriz bloco-diagonal (nN x nN)
    
    % 3. Monta a matriz bloco-diagonal das rotações
    R_big = diagcat(R_list{:}); 
    
    % 4. Exponencial e Produto
    R_pert_big = R_big * expm_casadi(S_big);
    
    % 5. Extração dos blocos de volta
    V = repmat(I_n, N, 1); 
    R_pert_all = R_pert_big * V;  % [R1*exp(S1); R2*exp(S2); ...; RN*exp(SN)]
    
    % 6. Substituição Única
    R_orig_all = vertcat(R_list{:});
    f_pert = substitute(f, R_orig_all, R_pert_all);
    
    % 7. Derivada Única e Avaliação em xi = 0
    J_expr = jacobian(f_pert, xi);
    J = substitute(J_expr, xi, zeros(total_vars, 1));
end