clc
clear all
import casadi.*

%% =========================================================
% Horizonte
%% =========================================================
N  = 5;
dt = 0.08;

v_min = -0.6;
v_max =  0.6;
x_sym = SX.sym('x_sym',6);
   v1 = x_sym(1);
   v2 = x_sym(2);
   omega = x_sym(3);
   a1 = x_sym(4);
   a2 = x_sym(5);
   alpha = x_sym(6);

S_sym = [0,-omega,v1,a1;omega,0,v2,a2;0,0,0,alpha;0,0,0,0];
omega_min = -0.8;
omega_max =  0.8;

%% =========================================================
% Variáveis de decisão
%% =========================================================
T_hor     = SX.sym('T',16,N+1);     % orientação (cos, sin)
T0     = SX.sym('T0',16,1); 
p     = SX.sym('p',2,N+1);     % posição
v     = SX.sym('v',2,N);       % velocidade linear
omega = SX.sym('omega',1,N);   % velocidade angular

%% =========================================================
% Referência
%% =========================================================
theta_ref = 0;

R_0 = [cos(30); sin(30)];
p_0 = [0; 0];
v_0 = [0; 0];
omega_0 = 0;
R_ref = [cos(theta_ref); sin(theta_ref)];
p_ref = [6; 5];

v_ref     = [0.0; 0.0];
omega_ref = 0;

%% =========================================================
% Matrizes SE(2)
%% =========================================================
T_ref = [R_ref(1), -R_ref(2), p_ref(1),v_ref(1);
         R_ref(2),  R_ref(1), p_ref(2),v_ref(2);
         0,         0,        1,omega_ref;
         0,         0,        0,         1];

 T_0 = [R_0(1), -R_0(2), p_0(1),v_0(1);
        R_0(2),  R_0(1), p_0(2),v_0(2);
        0,       0,     1,           omega_0;
        0,        0,    0,           1 ];

 custo = 0;
 Q =diag([2,2]);
 R = diag([1,1,1]);
 g = [];
 p_max = 9;
 p_min = -0.1;
  Matriz_T = reshape(T_hor(:,1), 4, 4);
  Matriz_T0 = reshape(T0, 4, 4);
  H_1 = Matriz_T(1:3,1:3);
 res0 =Matriz_T0(1:3,1:3)- H_1;
 h = [res0(:)];
 T_sym = cell(1,N+1);
  T_par = cell(1,N+2);
 T_par{1} = reshape(T0,4,4); 
for k =1:N
    
 Matriz_T = reshape(T_hor(:,k), 4, 4); 
L = (Matriz_T(1:2,3) - T_ref(1:2,3))'*Q*(Matriz_T(1:2,3) - T_ref(1:2,3))+(Matriz_T(1:3,4) - T_ref(1:3,4))'*R*(Matriz_T(1:3,4) - T_ref(1:3,4))+10*norm(eye(2)-eye(2)* Matriz_T(1:2,1:2),'fro')^2;
%  L = 100*norm(logm_casadi(inv(T_ref)*Matriz_T, 5, 3),'fro');
 
 
custo = custo+L;
Matriz_Tnext = reshape(T_hor(:,k+1), 4, 4);  
H_k = Matriz_T(1:3,1:3);
H_k_next = Matriz_Tnext(1:3,1:3);
omega_k =Matriz_T(3,4);
vx_k =Matriz_T(1,4); 
vy_k =Matriz_T(2,4); 
px_k =Matriz_T(1,3); 
py_k =Matriz_T(2,3);
Sk = [0,-omega_k,vx_k;omega_k,0,vy_k;0,0,0];
rest = H_k_next - H_k*(eye(3)+(dt*Sk));
rest = rest(1:2,1:3);
rest = rest(:);
h = [h; rest(3:end)];
g = [g;vx_k  - v_max; vy_k  - v_max;omega_k  - omega_max];
g = [g;-vx_k  + v_min; -vy_k  + v_min;omega_k  + omega_min];
g = [g;-px_k  + p_min; -py_k  + p_min];
g = [g;px_k  - p_max; py_k  - p_max];
T_sym{k} =  Matriz_T;
T_par{k+1} =  Matriz_T;
end
T_sym{k+1} = reshape(T_hor(:,k+1), 4, 4);
T_par{k+2} = reshape(T_hor(:,k+1), 4, 4);
reshape(T_hor(:,k), 4, 4);

[sym_data, casadi_funs] = solver_IPM(custo, T_sym,T0(:),S_sym,x_sym, g, h);



%% =========================================================
% Simulação em malha fechada (MPC)
%% =========================================================
Tsim = 1000; % número de passos

% Estado inicial real
T_real = full(T_0);

% Chute inicial do horizonte
T_guess = repmat(full(T_0(:)),1,N+1);

% Armazenamento
T_hist = zeros(16,Tsim);
vf  =[];
for t = 1:Tsim

    fprintf('Tempo %d\n', t);

    %% =====================================================
    % Resolver MPC
    %% =====================================================
    G0 = cell(N+1,1);
    for k = 1:N+1
        G0{k} = reshape(T_guess(:,k),4,4);
    end

    G_opt = run_optimization( ...
        sym_data, casadi_funs, G0, T_real(:), 200, 1e-5,1);

    %% =====================================================
    % Extrair solução ótima
    %% =====================================================
    T_opt = zeros(16,N+1);
    for k = 1:N+1
        var =G_opt{k}(:);
        T_opt(:,k) =full( var);
    end

    %% =====================================================
    % Aplicar primeiro controle (receding horizon)
    %% =====================================================
    T1 = reshape(T_opt(:,1),4,4);
    T2 = reshape(T_opt(:,2),4,4);

    % Extrair velocidades do modelo
    vx = T1(1,4)
    vy = T1(2,4)
    omega = T1(3,4)
     %% =====================================================
    % Modelo real (dinâmica)
    %% =====================================================
    S_real = [0,-omega,vx;
              omega,0,vy;
              0,0,0];

    H_real = T_real(1:3,1:3);

    % Integração (Euler)
    H_next = H_real*expm(dt*S_real)
   %  plot_robot_differential(H_next, dt);
    % Atualizar estado completo
    T_real(1:3,1:3) = H_next;
    T_real(1,4) = vx;
    T_real(2,4) = vy;
    T_real(3,4) = omega;

    %% =====================================================
    % Armazenar
    %% =====================================================
    T_hist(:,t) = T_real(:);

    %% =====================================================
    % Shift do horizonte (warm start)
    %% =====================================================
    T_guess(:,1:N) = T_opt(:,2:N+1);
    T_guess(:,N+1) = T_opt(:,N+1);

    % IMPORTANTE: atualizar condição inicial
    T_guess(:,1) = T_real(:);

end
