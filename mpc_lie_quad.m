clc
clear all
import casadi.*

%% =========================================================
% Parâmetros físicos
%% =========================================================
m = 2.24;
l = 0.332;
b = 9.5e-6;
Kt = 1.7e-7;
G = 9.81;
Ixx = 0.0213;
Iyy = 0.0363;
Izz = 0.0615;
I = diag([Ixx,Iyy,Izz]);

%% =========================================================
% Horizonte
%% =========================================================
N  = 10;
dt = 0.01;

%% =========================================================
% Variáveis simbólicas
%% =========================================================
x_sym = SX.sym('x_sym',16);
v = x_sym(1:3); 
a = x_sym(4:6); 
omega = x_sym(7:9);
a_omega = x_sym(10:12); 
u = x_sym(13:end); 
S_3 = [0,-omega(3),omega(2);omega(3),0,-omega(1);-omega(2),omega(1),0];
S_sym = [S_3,v,a,a_omega;zeros(4,6)] 
S_sym = [S_sym,[u;0;0;0]];
omega_max = 1;
omega_min = -1;
v_max = 5;
v_min = -5;
p_max = [15;15;15];
p_min = [-5;-5;-0.1];
umax = 12;
umin = 0;
T_hor = SX.sym('T',49,N+1);
T0    = SX.sym('T0',49,1);

Q = diag([50,50,50,1,1,1,1,1,1]);
R = diag([0.01,0.01,0.01,0.01]);

Rd = eye(3);

custo = 0;
g = [];
h = [];

%% =========================================================
% Restrição inicial
%% =========================================================
T_init = reshape(T_hor(:,0),7,7);
T0_mat = reshape(T0,7,7);
p_ref = [3;3;3];
res0 = T0_mat(1:6,1:6) - T_init(1:6,1:6);
h = [h; res0(:)];

%% =========================================================
% Loop do MPC
%% =========================================================
for k = 1:N

    T_k     = reshape(T_hor(:,k),7,7);
    T_next  = reshape(T_hor(:,k+1),7,7);

    H_k     = T_k(1:6,1:6);
    H_next  = T_next(1:6,1:6);

    Rk      = H_k(1:3,1:3);
    xi      = H_k(1:3,4);
    vk      = H_k(1:3,5);
    omega_k = H_k(1:3,6);

    u1_k = T_k(1:4,7);

    S_3k = [0,-omega_k(3),omega_k(2);
            omega_k(3),0,-omega_k(1);
            -omega_k(2),omega_k(1),0];

    xk = [xi-p_ref;vk;omega_k];

    %% custo
    u0 = [5.49;5.49;5.49;5.49];
    L = xk'*Q*xk + (u1_k- u0)'*R*(u1_k- u0) + norm(eye(3)-Rk'*Rd,'fro')^2;
    custo = custo + L;

    %% dinâmica
    c = 0.018;
    u = [1,1,1,1;
         0,l,0,-l;
         -l,0,l,0;
         c,-c,c,-c]*u1_k;

    T_force = [0;0;u(1)];
    ta = [u(2);u(3);u(4)];
    ay = [0;0;1];

    ak = (-G*ay + (1/m)*Rk*T_force);
    a_omega_k = inv(I)*(-S_3k*I*omega_k + ta);

    Sk = [S_3k, Rk'*vk, Rk'*ak, Rk'*a_omega_k;
          zeros(3,6)];

    %% restrição dinâmica
    rest = H_next - H_k*(eye(6) + dt*Sk);
    h = [h; rest(:)];

    %% restrições de estado
    g = [g;
         vk - v_max;
         -vk + v_min;
         omega_k - omega_max;
         -omega_k + omega_min;
         xi - p_max;
         -xi + p_min];
     g = [g;u1_k-umax];
     g = [g;-u1_k+umin];
     
T_sym{k} =  T_k;
end
T_sym{k+1} = reshape(T_hor(:,k+1), 7, 7);
reshape(T_hor(:,k), 7, 7);

[sym_data, casadi_funs] = solver_IPM(custo, T_sym,T0(:),S_sym,x_sym, g, h);


%% =========================================================
% Simulação MPC
%% =========================================================
Tsim = 200;

p0 = [0.1;0.1;0.1];
v0 = [0;0;0];
omega0 = [0;0;0];
u0 = [5.49;5.49;5.49;5.49];
R0 = eye(3);

T_0 = [R0,p0,v0,omega0;
       zeros(4,6)];
T_0 = [T_0,[u0;0;0;0]];

T_real = full(T_0);

T_guess = repmat(full(T_0(:)),1,N+1);

T_hist = zeros(49,Tsim);

%% =========================================================
% LOOP FECHADO (CORRIGIDO)
%% =========================================================
for t = 1:Tsim

    fprintf('Tempo %d\n', t);

    %% otimização
    G0 = cell(N+1,1);
    for k = 1:N+1
        G0{k} = reshape(T_guess(:,k),7,7);
    end

    G_opt = run_optimization(sym_data, casadi_funs, G0, T_real(:), 200, 1e-2, 1);

    %% solução
    T_opt = zeros(49,N+1);
    for k = 1:N+1
        T_opt(:,k) = full(G_opt{k}(:));
    end

    T1 = reshape(T_opt(:,1),7,7);

    %% =====================================================
    % MODELO REAL (IGUAL AO MPC)
    %% =====================================================
    Hk = T_real(1:6,1:6);

    Rk = Hk(1:3,1:3);
    xi = Hk(1:3,4);
    vk = Hk(1:3,5);
    omega_k = Hk(1:3,6);

    u1_k = T1(1:4,6);

    S_3k = [0,-omega_k(3),omega_k(2);
            omega_k(3),0,-omega_k(1);
            -omega_k(2),omega_k(1),0];

    c = 0.018;
    u = [1,1,1,1;
         0,l,0,-l;
         -l,0,l,0;
         c,-c,c,-c]*u1_k;

    T_force = [0;0;u(1)];
    ta = [u(2);u(3);u(4)];
    ay = [0;0;1];

    ak = (-G*ay + (1/m)*Rk*T_force);
    a_omega_k = inv(I)*(-S_3k*I*omega_k + ta);

    Sk = [S_3k, Rk'*vk, Rk'*ak, Rk'*a_omega_k;
          zeros(3,6)];

    %% integração
    H_next = Hk*(eye(6) + dt*Sk)

    %% atualizar estado
    T_real(1:6,1:6) = H_next;

    %% salvar
    T_hist(:,t) = T_real(:);

    %% shift horizonte
    T_guess(:,1:N) = T_opt(:,2:N+1);
    T_guess(:,N+1) = T_opt(:,N+1);
    T_guess(:,1) = T_real(:);

end