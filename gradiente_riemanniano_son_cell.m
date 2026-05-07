function [grad_normal, grad_cell] = gradiente_riemanniano_son_cell(f_sym, R_cell)
% GRADIENTE_RIEMANNIANO_SON_CELL_CASADI
% Gradiente Riemanniano para SO(n) usando CasADi (SX)
%
% Correção baseada no PDF (Seção 2.1) e geometria de SO(n)
%
% ENTRADAS:
%   f_sym  -> escalar SX
%   R_cell -> cell array com matrizes SX (n x n)
%
% SAÍDAS:
%   grad_normal -> vetor linha concatenado (1 x total_vars)
%   grad_cell   -> cell com gradientes (n x n)

    import casadi.*

    N = length(R_cell);
    n = size(R_cell{1}, 1);

    grad_cell = cell(N,1);
    grad_normal = SX([]);

    for k = 1:N

        Rk = R_cell{k};
        Rk_vec = reshape(Rk, n*n, 1);

        % =========================================================
        % 1) Gradiente Euclidiano (derivada em relação às entradas)
        % =========================================================
        grad_euc_vec = jacobian(f_sym, Rk_vec).';
        grad_euc = reshape(grad_euc_vec, n, n);

        % =========================================================
        % 2) Cálculo correto do gradiente Riemanniano para SO(n)
        % =========================================================
        % Para SO(n), o gradiente Riemanniano é dado por:
        % grad_R f = R * skew(R^T * grad_euc)
        % onde skew(A) = (A - A^T)/2
        
        % Calcular R^T * grad_euc
        Rtg = Rk' * grad_euc;
        
        % Parte anti-simétrica (projeção no espaço tangente em I)
        skew_part = (Rtg - Rtg') / 2;
        
        % Transportar para o ponto R (gradiente Riemanniano)
        grad_k = Rk * skew_part;

        % =========================================================
        % 3) Verificação: grad_k deve estar no espaço tangente
        %    Ou seja: Rk' * grad_k deve ser anti-simétrica
        % =========================================================
        % Opcional: verificação simbólica
        % Rk' * grad_k deve ser anti-simétrica
        
        % =========================================================
        % 4) Armazenar
        % =========================================================
        grad_cell{k} = grad_k;
        grad_normal = [grad_normal, reshape(grad_k, 1, n*n)];

    end

end