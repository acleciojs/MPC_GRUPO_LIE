
function [L2, H_sym, A_skew] = symbolic_Hessian( ...
    f, G_list, S_sym, omega_sym, tip)

    import casadi.*

    if nargin < 5 || isempty(tip)
        tip = 0;
    end

    if ~ismember(tip, [0 1 2])
        error('tip deve ser 0, 1 ou 2.');
    end

    N = numel(G_list);
    n = size(G_list{1}, 1);
    m = numel(omega_sym);
    total_vars = m * N;

    isSX = isa(f, 'casadi.SX');

    if isSX
        xi = SX.sym('xi', total_vars, 1);
    else
        xi = MX.sym('xi', total_vars, 1);
    end

    %% =====================================================
    % 1. Perturbacao global
    %% =====================================================

    R_pert_cell = cell(N, 1);

    for r = 1:N
        idx = (r-1)*m + (1:m);

        xi_r = xi(idx);

        S_r = substitute(S_sym, omega_sym, xi_r);

        R_pert_cell{r} = ...
            G_list{r} * expm_casadi(S_r);
    end

    f_pert_cell = substitute( ...
        {f}, G_list, R_pert_cell);

    f_pert = f_pert_cell{1};

    %% =====================================================
    % 2. Hessiana simetrica
    %% =====================================================

    H_expr = hessian(f_pert, xi);

    H_sym = substitute( ...
        H_expr, xi, zeros(total_vars, 1));

    if tip == 1
        L2 = H_sym;
        A_skew = [];
        return;
    end

    %% =====================================================
    % 3. Gradiente geometrico
    %% =====================================================

    J_expr = jacobian(f_pert, xi);

    L_full = substitute( ...
        J_expr, xi, zeros(total_vars, 1));

    %% =====================================================
    % 4. Base da algebra de Lie
    %% =====================================================

    E = jacobian( ...
        reshape(S_sym, n*n, 1), omega_sym);

    pinv_E = pinv(full(E));

    %% =====================================================
    % 5. Comutador simbolico global
    % SEM FOR i = 1:m
    %% =====================================================

    if isSX
        eta    = SX.sym('eta', m, 1);
        zeta   = SX.sym('zeta', m, 1);
        lambda = SX.sym('lambda', m, 1);
    else
        eta    = MX.sym('eta', m, 1);
        zeta   = MX.sym('zeta', m, 1);
        lambda = MX.sym('lambda', m, 1);
    end

    S_eta = substitute( ...
        S_sym, omega_sym, eta);

    S_zeta = substitute( ...
        S_sym, omega_sym, zeta);

    % Comutador de Lie
    comm = S_zeta*S_eta - S_eta*S_zeta;

    % Coordenadas do comutador na algebra
    coords = pinv_E * reshape(comm, n*n, 1);

    % Contracao com o gradiente generico
    scalar = lambda.' * coords;

    % Matriz antissimetrica generica
    % A(i,j) = -1/2 lambda' * c_ij

    A_template = -0.5 * jacobian( ...
        jacobian(scalar, zeta).', eta);

    %% =====================================================
    % 6. Montagem global
    % SEM FOR i = 1:m
    %% =====================================================

    if isSX
        A_skew = SX.zeros(total_vars, total_vars);
    else
        A_skew = MX.zeros(total_vars, total_vars);
    end

    for r = 1:N

        idx = (r-1)*m + (1:m);

        L_r = L_full(idx);

        % Substitui apenas o gradiente do bloco
        A_block = substitute( ...
            A_template, lambda, L_r.');

        A_skew(idx, idx) = A_block;

    end

    %% =====================================================
    % 7. Resultado
    %% =====================================================

    switch tip
        case 0
            % Hessiana geometrica completa
            L2 = H_sym + A_skew;

        case 2
            % Parte simetrica da Hessiana completa
            M = H_sym + A_skew;
            L2 = 0.5 * (M + M.');
    end

end
