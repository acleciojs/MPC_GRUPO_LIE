function mpc_quadrotor_artigo_lie_Tglobal_solverIPM()
clc; close all;
% MPC geometrico para quadrotor - Pereira, Leite e Raffo,
% "Stabilizing NMPC Approaches for Underactuated Mechanical Systems
% on the SE(3) Manifold", arXiv:2503.17458v2.
%
% Requer solver_IPM.m e run_optimization.m ORIGINAIS, CasADi e
% expm_casadi.m. Nesta versao solver_IPM NAO e modificado.
% Uma unica matriz T_G = blkdiag(T_0,...,T_N) forma T_sym={T_G}.
% Uma unica matriz S_G(omega) descreve a algebra do grupo produto.
% run_optimization recebe UMA matriz global e devolve UMA matriz global,
% que e desempacotada em N+1 blocos de 9x9 para uso do NMPC.
% mode = 'NMPC' ou 'FMNMPC'.
% NOVO: graficos ao vivo do horizonte MPC para Euler ZYX (roll,pitch,yaw),
%       erro geodesico de orientacao e historico real em malha fechada.
% REVISAO: eixo e3 no erro de atitude; ramo SO(3) exclui salto pi;
% verificacao de factibilidade; limites por rotor ativados com
% parametros de mistura ilustrativos (substitua por parametros reais).
import casadi.*

%% Configuracao
mode = 'NMPC'; % 'NMPC' ou 'FMNMPC'
N = 20; Ts = 0.01; Tsim = 1000;
max_iter = 200; tol = 1e-3; order = 0; reg = 1e-8;
m = 2.24; grav = 9.81;
J = diag([0.0363 0.0363 0.0615]);
e3 = [0;0;1];
% Pesos conforme secao 6.1 do artigo
kp = 150; kv = 30; kR = 10; kw = 0.85;
kf = 0.05; ktau = 0.3; zeta = 1.2;
k1 = 10; k3 = 1; % parametros k1 e k3 da secao 6.1
% O artigo usa limite por rotor 0 <= fi <= 12.3 N.
% Para aplicar limites individuais, fornecer l e ctf do veiculo.
use_rotor_limits = true; % fiel ao TIPO de restricao do artigo
l = 0.2; ctf = 0.01; % APENAS EXEMPLOS; SUBSTITUIR PELOS VALORES REAIS
Bmix = [1 1 1 1; 0 l 0 -l; -l 0 l 0; ctf -ctf ctf -ctf];
Tmax = 4*12.3; taumax = [2;2;2]; % usados apenas se rotor_limits=false
rot_branch_margin = 1e-3; % trace(E) >= -1+margin: exclui saltos de 180 graus
feas_tol = 0.9; % verificacao antes de aplicar controle
assert(l>0 && ctf>0,'l e ctf devem ser positivos e conhecidos.');
p_ref = [10;10;10]; R_ref = eye(3);
% Ajuste visual: atualiza o grafico do horizonte a cada iteracao.
plot_prediction_each_step = true;
plot_every = 1; % 1=todo passo MPC; 5=a cada cinco passos


%% Grupo PRODUTO: apenas UMA matriz T_global e UMA S_global
n = 9;
d = 16;
K = N+1;
nG = n*K;       % N=20 -> 189
D  = d*K;       % N=20 -> 336

% O UNICO elemento da tupla de grupos simbolicos:
T_global = SX.sym('T_global',nG,nG);
T_sym = {T_global};

% UMA algebra global, com 16 variaveis independentes por no.
omega_global = SX.sym('omega_global',D,1);
S_global = SX.zeros(nG,nG);
for k=1:K
    rows=(k-1)*n+(1:n);
    ids=(k-1)*d+(1:d);
    wk=omega_global(ids);
    Sk=SX.zeros(n,n);
    Sk(1:3,1:3)=hat3(wk(1:3));
    Sk(1:3,4:7)=reshape(wk(4:15),3,4);
    Sk(8,9)=wk(16);
    S_global(rows,rows)=Sk;
end

% Parametro externo = estado real no instante atual (9x9).
% O solver tem uma unica entrada de grupo, mas T0 continua com 81 valores.
T0 = SX.sym('T0',n*n,1);

% A expressao dos nos e extraida da UNICA matriz simbolica global.
T_node=cell(1,K);
for k=1:K
    rows=(k-1)*n+(1:n);
    T_node{k}=T_global(rows,rows);
end
assert(numel(T_sym)==1 && all(size(T_sym{1})==[nG nG]));
assert(all(size(S_global)==[nG nG]));

%% Condicao inicial: 3 equacoes locais de SO(3), mais p,v,w
Xinitial = reshape(T0,n,n);
Xfirst = T_node{1};
E0 = Xinitial(1:3,1:3)'*Xfirst(1:3,1:3);
h = [vee_skew(E0); Xfirst(1:3,4)-Xinitial(1:3,4); ...
     Xfirst(1:3,5)-Xinitial(1:3,5); ...
     Xfirst(1:3,6)-Xinitial(1:3,6)];
g = SX.zeros(0,1);
% vee_skew(E)=0 tambem em rotacoes de 180 graus.
% A desigualdade abaixo exclui esse ramo espurio da igualdade.
g = [g; rot_branch_margin - 1 - trace(E0)];
custo = SX(0);

%% Custo e restricoes do horizonte
for k=1:N
    Xk = T_node{k}; Xn = T_node{k+1};
    R = Xk(1:3,1:3); p = Xk(1:3,4);
    v = Xk(1:3,5); w = Xk(1:3,6);
    tau = Xk(1:3,7); thrust = Xk(8,9);
    Rn = Xn(1:3,1:3); pn = Xn(1:3,4);
    vn = Xn(1:3,5); wn = Xn(1:3,6);

    % Configuracao e1/e3 compativel com os ganhos k1 e k3
    Psi1 = 1 - (R(:,1)'*R_ref(:,1));
    Psi3 = 1 - (R(:,3)'*R_ref(:,3));
    Psi = k1*Psi1 + k3*Psi3;
    % Custo do artigo: kp||p||² + kv||v||² + kR*Psi² + kw||w||²
    Lstate = kp*sumsqr(p-p_ref) + kv*sumsqr(v) + ...
             kR*Psi^2 + kw*sumsqr(w);
    % O artigo usa norma da forca total inercial fI, inclusive gravidade.
    fI = thrust*R*e3-m*grav*e3;
    Linput = kf*sumsqr(fI) + ktau*sumsqr(tau);
    if strcmpi(mode,'FMNMPC')
        custo = custo + zeta^(k-1)*Lstate + Linput;
    else
        custo = custo + Lstate + Linput;
    end

    % Equacoes 11-15 do artigo
    p_pred = p + Ts*v;
    v_pred = v + (Ts/m)*fI;
    alpha = cross(J*w,w)+tau;
    wbar = w+(Ts/2)*(J\alpha);
    % Integracao exponencial: a predicao permanece exatamente em SO(3),
    % se R_k estiver no grupo. Nao usar Euler I+A ou projetar R depois.
    R_pred = R*(eye(size(R))+Ts*hat3(wbar));
    w_pred = wbar+(Ts/2)*(J\alpha);
    E = R_pred'*Rn;
    h = [h; pn-p_pred; vn-v_pred; wn-w_pred; vee_skew(E)];
    % vee_skew(E)=0 tem solucao espuria em rotacoes pi.
    % Ramo local valido: trace(E)>-1. Esta inequacao NAO e
    % uma restricao de ortogonalidade, e sim de desambiguacao do residual.
   % g = [g;rot_branch_margin - 1 - trace(E)];
      
    if use_rotor_limits
        fi = Bmix\[thrust;tau];
        g = [g;-fi;fi-12.3];
    else
        g = [g;-thrust;thrust-Tmax; tau-taumax; -tau-taumax];
    end
    % A desigualdade de ramo acima elimina a falsa igualdade de 180 graus.
  
end

% Controle terminal nao aparece na dinamica nem no custo:
% fixa-lo ao ultimo controle para evitar direcoes nulas na KKT.
Xn = T_node{N+1}; Xprev = T_node{N};
h = [h; Xn(1:3,7)-Xprev(1:3,7); Xn(8,9)-Xprev(8,9)];

fprintf(['solver_IPM ORIGINAL | G_sym = {T_global} | ' ...
         'T_global,S_global: %dx%d | d_global=%d | modo=%s\n'], ...
         nG,nG,D,mode);
% A interface do solver_IPM permanece INTACTA.
% solver_IPM ve N_groups=1, n=9*(N+1), d=16*(N+1).
% Seus symbolic_Jacobian e symbolic_Hessian operam no grupo produto.
assert(exist('solver_IPM','file')==2, 'solver_IPM.m nao encontrado');
assert(exist('run_optimization','file')==2, ...
       'run_optimization.m nao encontrado');
assert(exist('expm_casadi','file')==2, ...
       'expm_casadi.m necessaria para integracao SO(3) no MPC');
[sym_data,casadi_funs] = solver_IPM( ...
    custo,T_sym,T0(:),S_global,omega_global,g,h);
assert(sym_data.N==1 && sym_data.n==nG && sym_data.d==D);

%% Estado inicial e warm start
R0 = eye(3); p0 = [0;0;0]; v0 = zeros(3,1); w0 = zeros(3,1);
% Exemplo para testar a convergencia da orientacao (descomente):
R0 = eul_zyx_to_rotm([25;-15;40]*pi/180);

Xreal = packX(R0,p0,v0,w0,zeros(3,1),m*grav);
Xguess = cell(N+1,1);
for k=1:N+1, Xguess{k}=Xreal; end

%% Simulacao em malha fechada
p_hist = zeros(3,Tsim+1); v_hist = zeros(3,Tsim+1);
w_hist = zeros(3,Tsim+1); u_hist = zeros(4,Tsim);
R_hist = zeros(3,3,Tsim+1);
p_hist(:,1)=p0; v_hist(:,1)=v0; w_hist(:,1)=w0;
R_hist(:,:,1)=R0;

% Euler ZYX: linhas phi (roll), theta (pitch), psi (yaw)
euler_ref = rotm_to_euler_zyx(R_ref);       % [phi;theta;psi] rad
euler_hist = zeros(3,Tsim+1);
euler_hist(:,1) = rotm_to_euler_zyx(R0);
err_R_hist = nan(1,Tsim+1);                 % erro de SO(3), graus
err_R_hist(1) = rotation_error_deg(R_ref,R0);

% Um unico conjunto de graficos, atualizado SEM criar figuras repetidas.
jhor = 0:N;
if plot_prediction_each_step
    fig_pred = figure('Name','MPC - Orientacao ao longo do horizonte', ...
        'NumberTitle','off');
    h_pred = gobjects(4,1);
    h_ref = gobjects(3,1);
    h_real = gobjects(4,1);
    labels = {'\phi - Roll (graus)', ...
              '\theta - Pitch (graus)', ...
              '\psi - Yaw (graus)'};
    for ii=1:4
        ax = subplot(4,1,ii,'Parent',fig_pred);
        h_pred(ii) = plot(ax,jhor,nan(1,N+1),'b-', ...
            'LineWidth',1.8,'DisplayName','Previsao MPC');
        hold(ax,'on');
        if ii<=3
            h_ref(ii) = plot(ax,jhor, ...
                (180/pi)*euler_ref(ii)*ones(1,N+1),'k--', ...
                'LineWidth',1.2,'DisplayName','Referencia');
            h_real(ii) = plot(ax,0,nan,'ro','MarkerFaceColor','r', ...
                'DisplayName','Estado real atual');
            ylabel(ax,labels{ii});
        else
            plot(ax,jhor,zeros(1,N+1),'k--', ...
                'LineWidth',1.2,'DisplayName','Erro ideal (zero)');
            h_real(ii) = plot(ax,0,nan,'ro','MarkerFaceColor','r', ...
                'DisplayName','Erro real atual');
            ylabel(ax,'Erro SO(3) (graus)');
        end
        xlim(ax,[0 N]); grid(ax,'on');
        if ii==1 || ii==4
            legend(ax,'Location','best');
        end
        if ii==4
            xlabel(ax,'Passo futuro j no horizonte (0 ... N)');
        end
    end
    sgtitle(fig_pred,'Previsao de orientacao do MPC');
else
    fig_pred = [];
end
last = 0;
for t=1:Tsim
    tic;
    try
        % Warm start no UNICO elemento do grupo produto (189x189).
        % Nunca enviar os 21 nos separadamente ao run_optimization.
        Gguess = {blkdiag(Xguess{:})};
        Gopt = run_optimization(sym_data,casadi_funs, ...
            Gguess,Xreal(:),max_iter,tol,order,reg);
    catch ME
        warning('IPM falhou em t=%d: %s',t,ME.message);
        break
    end
    if ~iscell(Gopt) || numel(Gopt)~=1
        warning('Esperada UMA matriz de grupo global em t=%d.',t);
        break
    end
    Gmat=full(Gopt{1});
    if ~isequal(size(Gmat),[nG nG]) || any(~isfinite(Gmat(:)))
        warning('Matriz global invalida em t=%d.',t);
        break
    end
    % Extrair N+1 nos sem projetar ou normalizar as rotacoes.
    Xopt=cell(K,1);
    for kk=1:K
        ii=(kk-1)*n+(1:n);
        Xopt{kk}=Gmat(ii,ii);
    end
    [hmax,gmax,Rmax] = check_candidate(Xopt,Xreal,Ts,m,J,grav, ...
        use_rotor_limits,Bmix,Tmax,taumax,rot_branch_margin);
    % Euler ZYX e distancia SO(3) para TODOS os N+1 estados previstos.
    % Graficos sao feitos ANTES de aplicar u_0 a planta real.
    if plot_prediction_each_step && isgraphics(fig_pred) && ...
            mod(t-1,plot_every)==0
        eul_pred = zeros(3,N+1);
        rot_err_pred = zeros(1,N+1);
        for kk=1:N+1
            Rkk = Xopt{kk}(1:3,1:3);
            eul_pred(:,kk) = rotm_to_euler_zyx(Rkk);
            rot_err_pred(kk) = rotation_error_deg(R_ref,Rkk);
        end
        % Enrola o erro em [-pi,pi] e desembrulha ao longo do horizonte,
        % evitando saltos visuais de -180 a 180 graus.
        eul_diff = atan2(sin(eul_pred-euler_ref), ...
                         cos(eul_pred-euler_ref));
        eul_pred_plot = (180/pi)*(euler_ref+unwrap(eul_diff,[],2));
        % Medicao real em j=0, no mesmo ramo angular da referencia
        er_real = atan2(sin(euler_hist(:,t)-euler_ref), ...
                        cos(euler_hist(:,t)-euler_ref));
        eul_real_plot = (180/pi)*(euler_ref+er_real);
        for ii=1:3
            set(h_pred(ii),'XData',jhor,'YData',eul_pred_plot(ii,:));
            set(h_real(ii),'XData',0,'YData',eul_real_plot(ii));
        end
        set(h_pred(4),'XData',jhor,'YData',rot_err_pred);
        set(h_real(4),'XData',0,'YData',err_R_hist(t));
        sgtitle(fig_pred,sprintf( ...
            'MPC t=%d (%.2f s) | erro SO(3) real=%.2f graus | final previsto=%.2f graus', ...
            t,(t-1)*Ts,err_R_hist(t),rot_err_pred(end)));
        drawnow limitrate;
    end

    % Certificacao de viabilidade da solucao antes de aplicar ao VANT.
    % Nenhuma projecao altera Xopt; se falhar, nao aplicar o comando.
    if max([hmax,gmax,Rmax]) > feas_tol
        warning(['MPC inviavel em t=%d: |h|_inf=%.2e, ' ...
                 'max(g,0)=%.2e, SO3=%.2e; controle NAO aplicado.'], ...
                 t,hmax,gmax,Rmax);
      
    end
    thrust = Xopt{1}(8,9); tau = Xopt{1}(1:3,7);
    % Aplicar somente o primeiro controle a planta nao linear
    R = Xreal(1:3,1:3); p = Xreal(1:3,4);
    v = Xreal(1:3,5); w = Xreal(1:3,6);
    % Planta: RK4 para p,v,w, rotacao por exponencial em w medio
    [pn,vn,wn,Rn] = plant_step(R,p,v,w,thrust,tau,Ts,m,J,grav);
    Xreal = packX(Rn,pn,vn,wn,tau,thrust);
    p_hist(:,t+1)=pn; v_hist(:,t+1)=vn;
    w_hist(:,t+1)=wn; R_hist(:,:,t+1)=Rn;
    euler_hist(:,t+1)=rotm_to_euler_zyx(Rn);
    err_R_hist(t+1)=rotation_error_deg(R_ref,Rn);
    u_hist(:,t)=[thrust;tau];
    last=t;
    % Warm start com deslocamento do horizonte
    for k=1:N
        Xguess{k}=Xopt{k+1};
    end
    Xguess{N+1}=Xopt{N+1};
    Xguess{1}=Xreal;
    fprintf('t=%4d | ep=%.3e | ev=%.3e | T=%.3f | tau=%.3e | h=%.1e | %.2fs\n', ...
        t,norm(pn-p_ref),norm(vn),thrust,norm(tau),hmax,toc);
end

%% Graficos
if last==0, return; end
th=(0:last)*Ts; tu=(0:last-1)*Ts;
figure('Name','Quadrotor NMPC - artigo');
subplot(3,1,1);
plot(th,p_hist(:,1:last+1)','LineWidth',1.2); hold on;
yline(p_ref(3),'--'); grid on;
ylabel('Posicao (m)'); legend('x','y','z','z ref');
subplot(3,1,2);
plot(th,v_hist(:,1:last+1)','LineWidth',1.2); grid on;
ylabel('Velocidade (m/s)'); legend('vx','vy','vz');
subplot(3,1,3);
stairs(tu,u_hist(:,1:last)','LineWidth',1.2); grid on;
xlabel('Tempo (s)'); ylabel('Controle');
legend('T','tau_x','tau_y','tau_z');
figure('Name','Trajetoria 3D');
plot3(p_hist(1,1:last+1),p_hist(2,1:last+1), ...
      p_hist(3,1:last+1),'LineWidth',1.5);
grid on; axis equal; xlabel('x');ylabel('y');zlabel('z');

% HISTORICO REAL de Euler: o estado aplicado a planta em cada instante.
% Referencia constante. Use erro SO(3) para validar a convergencia mesmo
% com ambiguidade de Euler em theta=+/-90 graus.
figure('Name','Convergencia de Euler - malha fechada');
ang_hist_diff = atan2(sin(euler_hist(:,1:last+1)-euler_ref), ...
                      cos(euler_hist(:,1:last+1)-euler_ref));
ang_hist_deg = (180/pi)*(euler_ref+unwrap(ang_hist_diff,[],2));
ang_ref_deg = (180/pi)*euler_ref;
labels = {'\phi - Roll (graus)', ...
          '\theta - Pitch (graus)', ...
          '\psi - Yaw (graus)'};
for ii=1:3
    subplot(4,1,ii);
    plot(th,ang_hist_deg(ii,:),'b-','LineWidth',1.5); hold on;
    yline(ang_ref_deg(ii),'r--','LineWidth',1.2);
    ylabel(labels{ii}); grid on;
    legend('Real','Referencia','Location','best');
end
subplot(4,1,4);
plot(th,err_R_hist(1:last+1),'m-','LineWidth',1.5); hold on;
yline(0,'k--');
grid on; ylabel('Erro SO(3) (graus)'); xlabel('Tempo (s)');
sgtitle(sprintf('Erro final de orientacao = %.3f graus', ...
    err_R_hist(last+1)));
end

%% ========== INTERFACE ORIGINAL PRESERVADA ================
% Nenhum solver_IPM alternativo ou funcao local substituta e usado.
% Dependencias externas: solver_IPM, run_optimization, symbolic_Jacobian,
% symbolic_Hessian e expm_casadi.

function X=packX(R,p,v,w,tau,T)
X=eye(9);
X(1:3,1:3)=R; X(1:3,4)=p;
X(1:3,5)=v; X(1:3,6)=w;
X(1:3,7)=tau; X(8,9)=T;
end

function A=hat3(w)
A=[0 -w(3) w(2);w(3) 0 -w(1);-w(2) w(1) 0];
end

function z=vee_skew(E)
z=0.5*[E(3,2)-E(2,3);E(1,3)-E(3,1);E(2,1)-E(1,2)];
end

function [pn,vn,wn,Rn]=plant_step(R,p,v,w,T,tau,dt,m,J,g)
% Integracao RK4 das velocidades e posicao, com R no SO(3).
% Rotacao avaliada em estagios com exponenciais de w.
F=@(RR) (T/m)*RR(:,3)-g*[0;0;1];
W=@(ww) J\(tau-cross(ww,J*ww));
k1w=W(w); k1p=v; k1v=F(R);
w2=w+dt*k1w/2; R2=R*expm(hat3(dt*w/2));
k2w=W(w2); k2p=v+dt*k1v/2; k2v=F(R2);
w3=w+dt*k2w/2; R3=R*expm(hat3(dt*w2/2));
k3w=W(w3); k3p=v+dt*k2v/2; k3v=F(R3);
w4=w+dt*k3w; R4=R*expm(hat3(dt*w3));
k4w=W(w4); k4p=v+dt*k3v; k4v=F(R4);
pn=p+dt*(k1p+2*k2p+2*k3p+k4p)/6;
vn=v+dt*(k1v+2*k2v+2*k3v+k4v)/6;
wn=w+dt*(k1w+2*k2w+2*k3w+k4w)/6;
wavg=(w+2*w2+2*w3+w4)/6;
Rn=R*expm(hat3(dt*wavg));
end

function [hmax,gmax,Rmax]=check_candidate(Xopt,Xreal,Ts,m,J,grav, ...
     use_rotor_limits,Bmix,Tmax,taumax,branch)
% Calcula residuos PRIMAIS fisicos independentemente do IPM.
N=numel(Xopt)-1;
hmax=0; gmax=0; Rmax=0;
X0=Xopt{1};
E0=Xreal(1:3,1:3)'*X0(1:3,1:3);
hmax=max([hmax;abs(vee_skew(E0)); ...
    abs(X0(1:3,4)-Xreal(1:3,4)); ...
    abs(X0(1:3,5)-Xreal(1:3,5)); ...
    abs(X0(1:3,6)-Xreal(1:3,6))]);
gmax=max(gmax,branch-1-trace(E0));
for k=1:N+1
    X=Xopt{k}; R=X(1:3,1:3);
    Rmax=max(Rmax,max([norm(R'*R-eye(3),'fro');abs(det(R)-1)]));
    if k==N+1, continue; end
    Xn=Xopt{k+1};
    p=X(1:3,4);v=X(1:3,5);w=X(1:3,6);
    tau=X(1:3,7);T=X(8,9);
    fI=T*R(:,3)-m*grav*[0;0;1];
    alpha=cross(J*w,w)+tau;
    wbar=w+(Ts/2)*(J\alpha);
    % Igual ao MODELO do horizonte (exponencial, nao Cayley).
    Rp=R*expm(Ts*hat3(wbar));
    wp=wbar+(Ts/2)*(J\alpha);
    E=Rp'*Xn(1:3,1:3);
    hk=[Xn(1:3,4)-p-Ts*v; ...
        Xn(1:3,5)-v-Ts*fI/m; ...
        Xn(1:3,6)-wp;vee_skew(E)];
    hmax=max(hmax,max(abs(hk)));
    gmax=max(gmax,branch-1-trace(E));
    if use_rotor_limits
        fi=Bmix\[T;tau];
        gmax=max(gmax,max([-fi;fi-12.3]));
    else
        gmax=max(gmax,max([-T;T-Tmax;tau-taumax;-tau-taumax]));
    end
end
hmax=max([hmax; abs(Xopt{N+1}(1:3,7)-Xopt{N}(1:3,7)); ...
    abs(Xopt{N+1}(8,9)-Xopt{N}(8,9))]);
gmax=max(0,gmax);
end


function eul = rotm_to_euler_zyx(R)
% Converte matriz de rotacao R=Rz(psi)*Ry(theta)*Rx(phi).
% eul=[phi;theta;psi] em RADIANOS, sem toolbox.
% Gimbal lock: adotamos phi=0 quando |cos(theta)| fica pequeno.
    cy = hypot(R(1,1),R(2,1));
    theta = atan2(-R(3,1),cy);
    if cy > 1e-8
        phi = atan2(R(3,2),R(3,3));
        psi = atan2(R(2,1),R(1,1));
    else
        phi = 0;
        psi = atan2(-R(1,2),R(2,2));
    end
    eul = [phi;theta;psi];
end

function R = eul_zyx_to_rotm(eul)
% Auxiliar opcional para testar atitude inicial diferente de R_ref.
    phi=eul(1); theta=eul(2); psi=eul(3);
    c1=cos(phi); s1=sin(phi);
    c2=cos(theta); s2=sin(theta);
    c3=cos(psi); s3=sin(psi);
    R=[c3*c2, c3*s2*s1-s3*c1, c3*s2*c1+s3*s1; ...
       s3*c2, s3*s2*s1+c3*c1, s3*s2*c1-c3*s1; ...
          -s2,             c2*s1,             c2*c1];
end

function err_deg = rotation_error_deg(Rref,R)
% Distancia geodesica em SO(3) em [0,180] graus.
    c = (trace(Rref'*R)-1)/2;
    c = max(-1,min(1,c));
    err_deg = acos(c)*(180/pi);
end
