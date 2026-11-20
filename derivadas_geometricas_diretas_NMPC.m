function geo = derivadas_geometricas_diretas_NMPC(Ts,Q,Rcost,k1,k2, ...
                                      idx_bnd,x_min_b,x_max_b,u_min,u_max)
% DERIVADAS_GEOMETRICAS_DIRETAS_NMPC
% Gerador por estagio SEM CHAMAR A FUNCAO WRAPPER DE OTIMIZACAO.
%
% Chamadas diretas das rotinas ORIGINAIS do projeto:
%   symbolic_Jacobian(f,G_sym,S_sym,omega_sym)
%   symbolic_Hessian(L,G_sym,S_sym,omega_sym,0)
%
% Entrada do NMPC: x(30), u(8). Estado intrinseco q(20).
% Grupo de um no: G in R^(33x33), d=28 (20 estados + 8 controles).
% Estagio: Gk e Gn => 56 variaveis tangentes inicialmente, mas
% mantemos apenas [dq_k(20);dq_(k+1)(20);du_k(8)] = 48 variaveis.
%
% Hessiana: somente Hessiana geometrica do objetivo f (sem lambda,nu).
% Derivadas compiladas uma vez e reutilizadas nos 50 intervalos.
%
% Depende de symbolic_Jacobian.m e symbolic_Hessian.m disponibilizados
% pelo projeto, NAO incluidos neste pacote, e das funcoes do modelo.
import casadi.*

required = {'symbolic_Jacobian','symbolic_Hessian', ...
            'IntegrationEuler','Attitude_error','project_euler_state_symbolic'};
for j = 1:numel(required)
    assert(exist(required{j},'file')==2, ...
        'Dependencia ausente: %s.m (adicione ao path)',required{j});
end

nx=30; nq=20; nuu=8; d=28; n=33; nd=2*nq+nuu;
idx_bnd=idx_bnd(:); nb=numel(idx_bnd);
x_min_b=x_min_b(:); x_max_b=x_max_b(:);
u_min=u_min(:); u_max=u_max(:);
assert(all(idx_bnd>=1 & idx_bnd<=13), ...
    'Limites de estado neste gerador devem estar nos indices 1:13');
assert(numel(x_min_b)==nb && numel(x_max_b)==nb);
assert(numel(u_min)==nuu && numel(u_max)==nuu);

% Representacao de dois nos genericos na matriz de grupo.
Gk=SX.sym('Gk',n,n);
Gn=SX.sym('Gn',n,n);
G_sym={Gk,Gn};
Gvec=[Gk(:);Gn(:)];

param=SX.sym('ref',nx+nuu,1);  % [x_ref;u_ref]
xr=param(1:nx);
ur=param(nx+(1:nuu));

omega=SX.sym('omega',d,1);
S=lie_algebra_33(omega);

xk=unpack_group_state(Gk);
xnext=unpack_group_state(Gn);
uk=Gk(25:32,33);

%% Funcao objetivo: custo de um intervalo
f=cost_stage_original(xk,uk,xr,ur,Q,Rcost,k1,k2);

%% Igualdades de dinamica (13+3+4=20)
% A mesma expressao simbolica projetada e usada para derivar a QP.
F=project_euler_state_symbolic(IntegrationEuler(xk,uk,Ts));
F=F(:);
assert(numel(F)==nx,'IntegrationEuler precisa retornar 30 componentes');

E=reshape(F(14:22),3,3)'*reshape(xnext(14:22),3,3);
hR=0.5*[E(3,2)-E(2,3);E(1,3)-E(3,1);E(2,1)-E(1,2)];
h2=SX.zeros(4,1);
for j=0:3
    a=23+2*j;
    h2(j+1)=F(a)*xnext(a+1)-F(a+1)*xnext(a);
end
h=[xnext(1:13)-F(1:13);hR;h2];

%% Desigualdades de estado e controle
g=[xk(idx_bnd)-x_max_b; x_min_b-xk(idx_bnd); ...
   uk-u_max;u_min-uk];

%% Chamadas DIRETAS as funcoes de derivacao simbolica.
% Nao existe chamada a um wrapper intermediario.
grad_full=symbolic_Jacobian(f,G_sym,S,omega);
Jh_full=symbolic_Jacobian(h,G_sym,S,omega);
Jg_full=symbolic_Jacobian(g,G_sym,S,omega);
[H_full,~,~]=symbolic_Hessian(f,G_sym,S,omega,2);

% Ordem original do produto: [dq_k;du_k;dq_next;du_next].
% Ordem da SQP: [dq_k;dq_next;du_k]. Exclui du_next ficticio.
keep=[1:nq,d+(1:nq),nq+(1:nuu)];
assert(numel(keep)==nd && nd==48);

% Reducao somente das COLUNAS/COORDENADAS nao utilizadas.
% NAO e condensacao dinamica W'*H*W; sao submatrizes das derivadas
% de um estagio. Cada QP temporal continua nao condensada, 76 variaveis.
grad_full=grad_full(:);
grad_f=grad_full(keep);
Jh=Jh_full(:,keep);
Jg=Jg_full(:,keep);
Hf=H_full(keep,keep);

assert(numel(grad_f)==48 && size(Jh,2)==48);
assert(isequal(size(Hf),[48,48]),'Hessiana geometrica deve ser 48x48');
assert(size(Jh,1)==20 && numel(g)==2*nb+2*nuu);

%% Funcoes CasADi equivalentes as criadas anteriormente, sem wrapper.
cf=struct();
cf.f_fun=Function('nmpc_f_direct',{Gvec,param},{f});
cf.gradFun=Function('nmpc_grad_direct',{Gvec,param},{grad_f});
cf.HFun=Function('nmpc_Hf_direct',{Gvec,param},{Hf});
cf.h_fun=Function('nmpc_h_direct',{Gvec,param},{h});
cf.JhFun=Function('nmpc_Jh_direct',{Gvec,param},{Jh});
cf.g_fun=Function('nmpc_g_direct',{Gvec,param},{g});
cf.JgFun=Function('nmpc_Jg_direct',{Gvec,param},{Jg});
cf.S_fun=Function('nmpc_S_direct',{omega},{S});

geo.nx=nx;
geo.nx_t=nq;
geo.nq=nq;
geo.nu=nuu;
geo.nd=nd;
geo.ng=numel(g);
geo.keep=keep;
geo.casadi_funs=cf;
geo.lin=@(x,xn,u,xref,uref) ...
    evaluate_lin(cf,x,xn,u,xref,uref);
geo.hess=@(x,xn,u,xref,uref,varargin) ...
    evaluate_hess(cf,x,xn,u,xref,uref);
geo.bound_eval=@(x) evaluate_state_bounds(cf,x,nb);
geo.T_fun=@group_T_33;
geo.S_fun=@lie_algebra_33;

fprintf(['DERIVADAS DIRETAS compiladas: 2 grupos de 33x33, ' ...
         '48 incrementos ativos, %d igualdades, %d desigualdades.\n'], ...
         numel(h),numel(g));
end

function [gradf,Jh,hval,Jg,gval]=evaluate_lin(cf,x,xn,u,xref,uref)
GG=pack_pair(x,xn,u);
ref=[xref(:);uref(:)];
gradf=full(cf.gradFun(GG,ref));
Jh=full(cf.JhFun(GG,ref));
hval=full(cf.h_fun(GG,ref));
Jg=full(cf.JgFun(GG,ref));
gval=full(cf.g_fun(GG,ref));
gradf=gradf(:);
hval=hval(:);
gval=gval(:);
assert(isequal(size(Jh),[20 48]) && numel(gradf)==48,...
       'Dimensoes inesperadas das derivadas geometricas');
end

function H=evaluate_hess(cf,x,xn,u,xref,uref)
GG=pack_pair(x,xn,u);
ref=[xref(:);uref(:)];
H=full(cf.HFun(GG,ref));
assert(isequal(size(H),[48 48]),... 
       'Hessiana de estagio deve ter 48x48');
% Retorna Hessiana geometrica SOMENTE do custo f.
% varargin no handle geo.hess preserva compatibilidade com chamadas antigas
% que ainda enviavam lambda e nu; esses argumentos sao ignorados.
% Para qpoases, a simetrizacao/convexificacao ocorre em build da QP.
end

function [gb,Jgb]=evaluate_state_bounds(cf,x,nb)
GG=pack_pair(x,x,zeros(8,1));
ref=zeros(38,1);
gall=full(cf.g_fun(GG,ref));
Jall=full(cf.JgFun(GG,ref));
ngb=2*nb;
gb=gall(1:ngb);
Jgb=Jall(1:ngb,1:20);
end

function GG=pack_pair(x,xn,u)
G1=group_T_33(x,u);
G2=group_T_33(xn,zeros(8,1));
GG=[G1(:);G2(:)];
end

function G=group_T_33(x,u)
% A funcao aceita entradas numericas ou CasADi; aqui usada numericamente.
assert(numel(x)==30 && numel(u)==8);
G=eye(33);
G(1:3,1:3)=reshape(x(14:22),3,3);
for j=0:3
    b=23+2*j;
    ii=4+2*j;
    c=x(b); s=x(b+1);
    G(ii:ii+1,ii:ii+1)=[c,s;-s,c];
end
G(12:24,33)=x(1:13);
G(25:32,33)=u(:);
end

function x=unpack_group_state(G)
% Em G sao codificados somente os 30 estados fisicos.
xx=G(12:24,33);
R=reshape(G(1:3,1:3),9,1);
a=[G(4,4);G(4,5); G(6,6);G(6,7); ...
   G(8,8);G(8,9); G(10,10);G(10,11)];
x=[xx;R;a];
end

function S=lie_algebra_33(omega)
% A ordem dos incrementos por grupo e:
% [dx_e(13); dphi_SO3(3); dtheta_SO2(4); du(8)].
import casadi.*
S=SX.zeros(33,33);
v=omega(14:16);
S(1:3,1:3)=[0,-v(3),v(2);v(3),0,-v(1);-v(2),v(1),0];
for j=0:3
    a=omega(17+j); ii=4+2*j;
    S(ii:ii+1,ii:ii+1)=[0,a;-a,0];
end
S(12:24,33)=omega(1:13);
S(25:32,33)=omega(21:28);
end

function y=cost_stage_original(x,u,xr,ur,Q,Rc,k1,k2)
% Identico a cost_stage_sym do codigo original fornecido.
phiB=Attitude_error(x(14:22),xr(14:22),k1,k2);
phi1=Attitude_error(x(23:24),xr(23:24),k1,k2);
phi2=Attitude_error(x(25:26),xr(25:26),k1,k2);
phi3=Attitude_error(x(27:28),xr(27:28),k1,k2);
phi4=Attitude_error(x(29:30),xr(29:30),k1,k2);
e=[x(1:13)-xr(1:13);phiB;phi1;phi2;phi3;phi4];
y=e'*Q*e+(u-ur)'*Rc*(u-ur);
end
