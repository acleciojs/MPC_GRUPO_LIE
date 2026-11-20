function geo = solver_IPM_stage_NMPC(Ts,Q,Rcost,k1,k2, ...
                                       idx_bnd,x_min_b,x_max_b,u_min,u_max)
% SOLVER_IPM_STAGE_NMPC  Adaptador REAL do solver_IPM do usuario.
%
% Compila UMA vez o problema de UM intervalo [k,k+1]; no NMPC a
% mesma representacao e reutilizada nos N intervalos do horizonte.
%
% x fisico = [xe(13); vec(R_body)(9); quatro pares (R11,R13)(8)] (30)
% u fisico = 8; incremento tangente de x = 20; de (x,u) = 28.
%
% Matriz de grupo G(33x33) = diag(R,SO2_1,...,SO2_4,I13,I8,1)
% com xe e u na ultima coluna (grupo produto, subgrupo de SL(33)).
% A saida de solver_IPM tem derivadas para DOIS grupos Gk,Gk+1,
% portanto tem 56 coordenadas; retiramos as 8 coordenadas de u_(k+1),
% que NAO sao variaveis da formulacao de estagio.
%
% A Hessiana HFun e Hessiana da Lagrangiana f+lambda'*h+nu'*g.
% O termo linear da QP vem exclusivamente de gradFun (grad f).
%
% DEPENDENCIAS EXTERNAS obrigatorias:
% solver_IPM.m, symbolic_Jacobian.m, symbolic_Hessian.m,
% IntegrationEuler.m, Attitude_error.m, CasADi.
% Nao usa derivador alternativo nem substitui solver_IPM.
import casadi.*

required = {'solver_IPM','symbolic_Jacobian','symbolic_Hessian', ...
            'IntegrationEuler','Attitude_error'};
for j=1:numel(required)
    assert(exist(required{j},'file')==2, ...
        'Dependencia ausente: %s.m (adicione ao path)',required{j});
end

nx=30; nq=20; nuu=8; d=28; n=33; nd=2*nq+nuu;
idx_bnd=idx_bnd(:); nb=numel(idx_bnd);
x_min_b=x_min_b(:); x_max_b=x_max_b(:);
u_min=u_min(:); u_max=u_max(:);
assert(all(idx_bnd>=1 & idx_bnd<=13), ...
    'Limites de estado neste adaptador devem estar nos indices 1:13');
assert(numel(x_min_b)==nb && numel(x_max_b)==nb);
assert(numel(u_min)==nuu && numel(u_max)==nuu);

% Matrizes totalmente simbolicas para que os mecanismos de substituicao
% dentro de symbolic_Jacobian/Hessian possam perturbar Gk e Gk+1.
Gk=SX.sym('Gk',n,n);
Gn=SX.sym('Gn',n,n);
G_sym={Gk,Gn};
param=SX.sym('ref',nx+nuu,1);    % [x_ref;u_ref]
xr=param(1:nx);
ur=param(nx+(1:nuu));
omega=SX.sym('omega',d,1);
S=lie_algebra_33(omega);

xk=unpack_group_state(Gk);
xnext=unpack_group_state(Gn);
uk=Gk(25:32,33);

f=cost_stage_original(xk,uk,xr,ur,Q,Rcost,k1,k2);
F=IntegrationEuler(xk,uk,Ts);
F=F(:);
assert(numel(F)==nx,'IntegrationEuler precisa retornar 30 componentes');

% Igualdades de dinamica: 13 euclidianas + 3 em so(3) + 4 em so(2).
% Residuos LOCAIS; podem ter ramo espurio a 180 graus.
E=reshape(F(14:22),3,3)' * reshape(xnext(14:22),3,3);
h_R=0.5*[E(3,2)-E(2,3);E(1,3)-E(3,1);E(2,1)-E(1,2)];
h_j=SX.zeros(4,1);
for j=0:3
    a=23+2*j;
    h_j(j+1)=F(a)*xnext(a+1)-F(a+1)*xnext(a);
end
h=[xnext(1:13)-F(1:13); h_R; h_j];

g=[xk(idx_bnd)-x_max_b; x_min_b-xk(idx_bnd); ...
   uk-u_max; u_min-uk];

% Esta E A CHAMADA AO METODO ORIGINAL DO USUARIO.
[sym_data,casadi_funs]=solver_IPM( ...
    f,G_sym,param,S,omega,g,h);

assert(sym_data.N==2 && sym_data.n==33 && sym_data.d==28, ...
       'solver_IPM: dimensoes inesperadas para o grupo produto');
assert(sym_data.p==20 && sym_data.m==2*nb+2*nuu, ...
       'solver_IPM: numero de restricoes inesperado');

% Permutacao de [dq_k;du_k;dq_next;du_next] para a ordem
% [dq_k;dq_next;du_k] usada pela SQP local. As ultimas 8 colunas
% du_next NAO participam, eliminando o controle terminal ficticio.
keep=[1:nq, d+(1:nq), nq+(1:nuu)];
assert(numel(keep)==nd && nd==48);

geo.nx=nx;
geo.nx_t=nq;
geo.nq=nq;
geo.nu=nuu;
geo.nd=nd;
geo.ng=sym_data.m;
geo.idx_bnd=idx_bnd;
geo.sym_data=sym_data;
geo.casadi_funs=casadi_funs;
geo.keep=keep;
geo.lin=@(x,xn,u,xref,uref) ...
    evaluate_lin(casadi_funs,x,xn,u,xref,uref,keep);
geo.hess=@(x,xn,u,xref,uref,lambda,nu_dual) ...
    evaluate_hess(casadi_funs,x,xn,u,xref,uref,lambda,nu_dual,keep);
geo.bound_eval=@(x) ...
    evaluate_state_bounds(casadi_funs,x,nb,keep);
geo.T_fun=@group_T_33;
geo.S_fun=@lie_algebra_33;

fprintf(['solver_IPM ORIGINAL conectado: N_local=2, grupo=33x33, ' ...
         'd_grupo=28, eta_SQP=48, h=20, g=%d.\n'],geo.ng);
end

function [gradf,Jh,hval,Jg,gval]=evaluate_lin( ...
                                  cf,x,xn,u,xref,uref,keep)
GG=pack_pair(x,xn,u);
param=[xref(:);uref(:)];
grad_all=full(cf.gradFun(GG,param));
Jh_all=full(cf.JhFun(GG,param));
hval=full(cf.h_fun(GG,param));
hval=hval(:);
Jg_all=full(cf.JgFun(GG,param));
gval=full(cf.g_fun(GG,param));
gval=gval(:);

grad_all=grad_all(:);
gradf=grad_all(keep);
Jh=Jh_all(:,keep);
Jg=Jg_all(:,keep);
assert(isequal(size(Jh),[20 48]) && numel(gradf)==48, ...
       'Dimensao inesperada de derivadas do solver_IPM');
end

function H=evaluate_hess(cf,x,xn,u,xref,uref,lambda,nu_dual,keep)
GG=pack_pair(x,xn,u);
param=[xref(:);uref(:)];
HL_all=full(cf.HFun(GG,param,lambda(:),nu_dual(:)));
assert(isequal(size(HL_all),[56 56]), ...
       'A HFun precisa retornar uma matriz 56x56');
H=HL_all(keep,keep);
% O termo quadratico utiliza obrigatoriamente a parte simetrica.
H=0.5*(H+H');
end

function [gb,Jgb]=evaluate_state_bounds(cf,x,nb,keep)
% Usa DIRETAMENTE g_fun e JgFun produzidas pelo solver_IPM.
% Controle e referencia ficticios: g de estado independe de ambos.
GG=pack_pair(x,x,zeros(8,1));
param=zeros(38,1);
gall=full(cf.g_fun(GG,param));
Jall=full(cf.JgFun(GG,param));
ngb=2*nb;
gb=gall(1:ngb);
Jgb=Jall(1:ngb,keep(1:20));
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
