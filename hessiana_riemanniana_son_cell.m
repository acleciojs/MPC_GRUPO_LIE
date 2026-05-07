function H = hessiana_riemanniana_son_cell(f_sym, R_cell)
% HESSIANA_RIEMANNIANA_SON_CELL
% Hessiana Riemanniana para SO(n) usando CasADi (SX)
%
% Baseada na definição do PDF:
% Hess(f)(X,Y) = ⟨∇_X(grad f), Y⟩
%
% Para SO(n) com métrica bi-invariante, a conexão de Levi-Civita é:
% ∇_η ξ = R * skew( R^T * dξ(η) ) + η * skew( R^T * ξ )
% onde dξ(η) é a derivada direcional de ξ na direção η
%
% ENTRADAS:
%   f_sym   -> escalar SX
%   R_cell  -> cell array com matrizes SX (n x n)
%
% SAÍDA:
%   H       -> matriz Hessiana (dim_total x dim_total)

    import casadi.*

    N = length(R_cell);
    n = size(R_cell{1}, 1);
    dim = n * n;
    dim_total = N * dim;
    
    H = SX.zeros(dim_total, dim_total);
    
    % =========================================================
    % 1) Construir vetor de variáveis concatenado
    % =========================================================
    R_all = [];
    for k = 1:N
        R_all = [R_all; reshape(R_cell{k}, dim, 1)];
    end
    
    % =========================================================
    % 2) Calcular gradiente Riemanniano de forma simbólica
    % =========================================================
    % Primeiro, calcular gradiente Euclidiano
    grad_euc_all = jacobian(f_sym, R_all).';  % vetor coluna dim_total x 1
    
    % Reorganizar em células
    grad_euc_cell = cell(N, 1);
    grad_riem_cell = cell(N, 1);
    
    for k = 1:N
        idx = (k-1)*dim + 1 : k*dim;
        grad_euc_k = reshape(grad_euc_all(idx), n, n);
        grad_euc_cell{k} = grad_euc_k;
        
        % Gradiente Riemanniano no ponto R_k
        Rk = R_cell{k};
        grad_riem_cell{k} = grad_riem_so_n(Rk, grad_euc_k);
    end
    
    % =========================================================
    % 3) Construir Hessiana via diferenciação do gradiente
    % =========================================================
    % A Hessiana Riemanniana é a derivada covariante do gradiente:
    % Para cada direção, precisamos da derivada do gradiente projetada
    
    % Vetor de gradientes Riemannianos concatenado
    grad_riem_vec = [];
    for k = 1:N
        grad_riem_vec = [grad_riem_vec; reshape(grad_riem_cell{k}, dim, 1)];
    end
    
    % Jacobiano do gradiente Riemanniano em relação às variáveis
    % Isso dá a derivada Euclidiana, que precisa ser corrigida
    J_riem = jacobian(grad_riem_vec, R_all);  % dim_total x dim_total
    
    % =========================================================
    % 4) Corrigir para obter a derivada covariante
    % =========================================================
    % A Hessiana Riemanniana H satisfaz:
    % H * v = ∇_v (grad f) = P_R( d(grad f)[v] )
    % onde P_R é a projeção no espaço tangente em R
    
    for j = 1:N  % coluna: direção da derivada
        Rj = R_cell{j};
        idx_j = (j-1)*dim + 1 : j*dim;
        
        for i = 1:N  % linha: componente do gradiente
            Ri = R_cell{i};
            idx_i = (i-1)*dim + 1 : i*dim;
            
            if i == j
                % =============================================
                % Bloco diagonal: ∇_{η} (grad_i f)
                % =============================================
                % Para cada direção base, calcular a derivada covariante
                H_block = SX.zeros(dim, dim);
                
                for col = 1:dim
                    % Direção base (vetor tangente em Rj)
                    eta_base = SX.zeros(dim, 1);
                    eta_base(col) = 1;
                    
                    % Derivada Euclidiana do gradiente nesta direção
                    d_grad_euc = SX.zeros(n, n);
                    for k = 1:dim
                        d_grad_euc = d_grad_euc + ...
                            reshape(J_riem(idx_i, idx_j(k)) * eta_base(k), n, n);
                    end
                    
                    % Derivada covariante: ∇_η (grad f) = R * skew( R^T * d_grad_euc )
                    Hcol = Ri * skew(Ri' * d_grad_euc);
                    
                    % Adicionar termo de conexão (importante!)
                    % Para métrica bi-invariante, a conexão tem um termo adicional
                    eta = reshape(eta_base, n, n);
                    termo_conexao = eta * skew(Ri' * grad_euc_cell{i});
                    Hcol = Hcol + termo_conexao;
                    
                    % Projetar no espaço tangente (garantia)
                    Hcol = proj_tangente_so_n(Hcol, Ri);
                    
                    H_block(:, col) = reshape(Hcol, dim, 1);
                end
                
                H(idx_i, idx_j) = H_block;
                
            else
                % =============================================
                % Bloco fora da diagonal: derivada mista
                % =============================================
                % Para i ≠ j, a Hessiana é simplesmente a derivada do
                % gradiente de i em relação a Rj (sem correção de conexão)
                H(idx_i, idx_j) = J_riem(idx_i, idx_j);
            end
        end
    end
    
    % =========================================================
    % 5) Garantir simetria (opcional)
    % =========================================================
    H = (H + H') / 2;
end

% ======================================================================
% Funções auxiliares
% ======================================================================

function grad_riem = grad_riem_so_n(R, grad_euc)
% Gradiente Riemanniano para SO(n)
    n = size(R, 1);
    Rtg = R' * grad_euc;
    skew_part = (Rtg - Rtg') / 2;
    grad_riem = R * skew_part;
end

function A_skew = skew(A)
% Parte anti-simétrica de uma matriz
    A_skew = (A - A') / 2;
end

function V_proj = proj_tangente_so_n(V, R)
% Projeção no espaço tangente de SO(n) no ponto R
    RTV = R' * V;
    skew_part = (RTV - RTV') / 2;
    V_proj = R * skew_part;
end