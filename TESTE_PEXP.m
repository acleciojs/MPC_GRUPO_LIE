clc; clear; close all;

%% =========================================================
% Parâmetros
%% =========================================================
omega = 1.0;              % velocidade angular constante
T     = 2.0;              % tempo total
dt_vec = logspace(-4, 0, 100);  % vários ?t

err = zeros(size(dt_vec));

%% =========================================================
% Loop sobre ?t
%% =========================================================
for k = 1:length(dt_vec)
    
    dt = dt_vec(k);
    N  = round(T/dt);
    
    % Matriz Omega (so(2))
    Omega = [0 -omega;
             omega 0];
    
    % Inicialização
    R_exp  = eye(2);  % solução exata (expm)
    R_euler = eye(2); % aproximação 1ª ordem
    
    %% =====================================================
    % Integração
    %% =====================================================
    for i = 1:N
        
        % Exata
        R_exp = R_exp * expm(Omega*dt);
        
        % Primeira ordem
        R_euler = R_euler * (eye(2) + Omega*dt);
    end
    
    %% =====================================================
    % Erro (norma matricial)
    %% =====================================================
    err(k) = norm(R_exp - R_euler, 'fro');
end

%% =========================================================
% Plot
%% =========================================================
figure;
loglog(dt_vec, err, 'o-','LineWidth',2);
grid on;
xlabel('\Deltat');
ylabel('Erro (norma Frobenius)');
title('Erro da integração exponencial de 1ª ordem');

% referência O(dt)
hold on;
loglog(dt_vec, dt_vec, '--k');
legend('Erro','O(\Deltat)');