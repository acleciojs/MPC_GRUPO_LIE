function out = exemplo_sqp_lie_admm_SO2()
% EXEMPLO FUNCIONAL:  NMPC simplificado de robô planar sobre SO(2).
% Modelo de horizonte com N=6 intervalos, M=3 blocos temporais,
% n_delta=33 coordenadas tangentes; igualdade dinamica nao linear,
% desigualdades de controle e incremento exponencial em SO(2).
%
% Requer: CasADi MATLAB, qpOASES e arquivos da pasta.
% Para usar qpOASES em vez de qpOASES: opts.qp_solver='qpoases'.
import casadi.*
N=6;M=3;bs=N/M;dt=0.1;
nx=3;nu=2;          % estado tangente (px,py,theta), controle (v,w)
na=6;               % estado ambiente (px,py,vec(R2))
nd=nx*(N+1)+nu*N;
np=na*(N+1)+nu*N;
X=SX.sym('X',np,1);
d=SX.sym('delta',nd,1);
J=[0 -1;1 0];

% Indices: estado ambiente [p_x,p_y,R(:)] e tangente [dp_x,dp_y,dtheta].
ix=cell(N+1,1);id=cell(N+1,1);
for k=1:N+1
    ix{k}=na*(k-1)+(1:na);
    id{k}=nx*(k-1)+(1:nx);
end
ixu=cell(N,1);idu=cell(N,1);
for k=1:N
    ixu{k}=na*(N+1)+nu*(k-1)+(1:nu);
    idu{k}=nx*(N+1)+nu*(k-1)+(1:nu);
end

p=cell(N+1,1);R=cell(N+1,1);u=cell(N,1);
for k=1:N+1
    ii=ix{k};jj=id{k};
    p{k}=X(ii(1:2))+d(jj(1:2));
    a=d(jj(3));
    Rot=[cos(a),-sin(a);sin(a),cos(a)];
    R{k}=reshape(X(ii(3:6)),2,2)*Rot;
end
for k=1:N
    u{k}=X(ixu{k})+d(idu{k});
end

Rref=[cos(0.25),-sin(0.25);sin(0.25),cos(0.25)];
pref=[0.65;0.25];
uref=[0.50;0.00];
f=SX(0);g=SX.zeros(0,1);h=SX.zeros(0,1);
for k=1:N
    pk=p{k}; Rk=R{k}; Rnext=R{k+1}; uk=u{k};
    ep=pk-pref;
    Eref=Rref.'*Rk;
    er=2-Eref(1,1)-Eref(2,2);
    eu=uk-uref;
    % Custo apenas de estagio; sem custo terminal.
    f=f+20*(ep.'*ep)+4*er+0.25*(eu.'*eu);

    % Dinamica discreta: p+ = p + dt*v*R*e1.
    gpos=p{k+1}-pk-dt*uk(1)*Rk(:,1);
    % Residuo SO(2): localmente equivalente a
    % R(k+1) = R(k) exp(dt*w*J). Valido proximo a identidade.
    a=-dt*uk(2);
    RotNeg=[cos(a),-sin(a);sin(a),cos(a)];
    E=Rk.'*Rnext*RotNeg;
    grot=0.5*(E(2,1)-E(1,2));
    g=[g;gpos;grot];

    % Exemplo de desigualdades h(x)<=0.
    h=[h; -uk(1); uk(1)-1.2; -uk(2)-0.8; uk(2)-0.8];
end

[H,~]=hessian(f,d);
Jg=jacobian(g,d);
Jh=jacobian(h,d);

% Condicao nominal cinematicamente consistente (linha reta).
X0=zeros(np,1);
for k=1:N+1
    ii=ix{k};
    X0(ii(1:2))=[0.5*(k-1)*dt;0];
    X0(ii(3:6))=reshape(eye(2),4,1);
end
for k=1:N,X0(ixu{k})=uref;end

% Configuracao do passo EXPONENCIAL EXTERNO.
exp_cfg=struct();exp_cfg.euclid_x_idx=[];exp_cfg.euclid_d_idx=[];
for k=1:N+1
    ii=ix{k};jj=id{k};
    exp_cfg.euclid_x_idx=[exp_cfg.euclid_x_idx,ii(1:2)]; %#ok<AGROW>
    exp_cfg.euclid_d_idx=[exp_cfg.euclid_d_idx,jj(1:2)]; %#ok<AGROW>
    exp_cfg.groups(k).x_idx=ii(3:6); %#ok<AGROW>
    exp_cfg.groups(k).delta_idx=jj(3); %#ok<AGROW>
    exp_cfg.groups(k).basis=J; %#ok<AGROW>
end
for k=1:N
    exp_cfg.euclid_x_idx=[exp_cfg.euclid_x_idx,ixu{k}]; %#ok<AGROW>
    exp_cfg.euclid_d_idx=[exp_cfg.euclid_d_idx,idu{k}]; %#ok<AGROW>
end

% M=3 QPs, compartilhando estados nas fronteiras k=3,5.
indices_blocos=cell(M,1);
for i=1:M
    k0=(i-1)*bs;
    ids=[];
    for k=k0+1:k0+bs+1
        ids=[ids,id{k}]; %#ok<AGROW>
    end
    for k=k0+1:k0+bs
        ids=[ids,idu{k}]; %#ok<AGROW>
    end
    indices_blocos{i}=ids;
end

opts=struct();opts.max_sqp=1;
% 1200 iteracoes foram insuficientes no teste relatado (r/eps=3.22).
% Aumentar apenas o orcamento nao muda as atualizacoes do ADMM.
opts.rho=10;opts.max_admm=2500;
opts.tol_abs=1e-5;opts.tol_rel=1e-5;
opts.reg_base=1e-4;opts.reg_floor=1e-5;
opts.qp_solver='qpoases';
opts.compare_central=true;
opts.verbose=true;opts.verbose_admm=true;opts.verbose_every=100;

% Corrigir estado inicial: delta(p0) = 0; delta(theta0) = 0.
opts.lb_delta=-inf(nd,1);opts.ub_delta=inf(nd,1);
opts.lb_delta(id{1})=0;opts.ub_delta(id{1})=0;

out=sqp_lie_admm(f,H,g,Jg,h,Jh,d,X,X0,indices_blocos,exp_cfg,opts);
fprintf('\nStatus: %s | SQP aplicadas: %d\n',out.status,out.completed_sqp);
if ~isempty(out.sqp)
    fprintf('ADMM: %s | iteracoes=%d\n', ...
        out.sqp(end).admm.reason,out.sqp(end).admm.iter);
end
if ~isempty(out.central_comparison)
    fprintf('Erro relativo ADMM/central=%.3e\n', ...
        out.central_comparison(end).erro_relativo);
end
end
