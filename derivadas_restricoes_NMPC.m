function geo = derivadas_restricoes_NMPC(Ts,idx_bnd,x_min_b,x_max_b,u_min,u_max)
% DERIVADAS_RESTRICOES_NMPC
% Deriva APENAS h e g. Hessiana e gradiente simetricos de f sao GLOBAIS
% e calculados
% em derivadas_globais_horizonte_NMPC. Nao chama symbolic_Hessian.
% Dois nos genericos Gk,Gn in R^(33x33); 48 coordenadas ativas:
% eta=[dq_k(20);dq_next(20);du_k(8)].
import casadi.*
required={'symbolic_Jacobian','IntegrationEuler','project_euler_state_symbolic'};
for j=1:numel(required)
    assert(exist(required{j},'file')==2,'Dependencia ausente: %s.m',required{j});
end
idx_bnd=idx_bnd(:); x_min_b=x_min_b(:); x_max_b=x_max_b(:);
u_min=u_min(:); u_max=u_max(:);
assert(all(idx_bnd>=1 & idx_bnd<=13));
assert(numel(x_min_b)==numel(idx_bnd) && numel(x_max_b)==numel(idx_bnd));
assert(numel(u_min)==8 && numel(u_max)==8);

Gk=SX.sym('G_din_k',33,33);
Gn=SX.sym('G_din_n',33,33);
G_sym={Gk,Gn};
GG=[Gk(:);Gn(:)];
w=SX.sym('omega_dyn',28,1);
S=lie_algebra_33_restr(w);

x=unpack_group_state_restr(Gk);
xn=unpack_group_state_restr(Gn);
u=Gk(25:32,33);
F=project_euler_state_symbolic(IntegrationEuler(x,u,Ts));
F=F(:);
assert(numel(F)==30,'IntegrationEuler deve fornecer 30 estados');
Rerr=reshape(F(14:22),3,3)'*reshape(xn(14:22),3,3);
hR=0.5*[Rerr(3,2)-Rerr(2,3); ...
         Rerr(1,3)-Rerr(3,1); ...
         Rerr(2,1)-Rerr(1,2)];
h2=SX.zeros(4,1);
for j=0:3
    a=23+2*j;
    h2(j+1)=F(a)*xn(a+1)-F(a+1)*xn(a);
end
h=[xn(1:13)-F(1:13);hR;h2];
g=[x(idx_bnd)-x_max_b; x_min_b-x(idx_bnd); ...
   u-u_max;u_min-u];
Jh_all=symbolic_Jacobian(h,G_sym,S,w);
Jg_all=symbolic_Jacobian(g,G_sym,S,w);
% Ordenacao de dois grupos: [dq_k;du_k;dq_next;du_next].
keep=[1:20, 28+(1:20), 20+(1:8)];
Jh=Jh_all(:,keep);
Jg=Jg_all(:,keep);
assert(isequal(size(Jh),[20 48]));
cf.h=Function('NMPC_h_restr',{GG},{h});
cf.Jh=Function('NMPC_Jh_restr',{GG},{Jh});
cf.g=Function('NMPC_g_restr',{GG},{g});
cf.Jg=Function('NMPC_Jg_restr',{GG},{Jg});
geo.nx_t=20; geo.nq=20; geo.nu=8; geo.nd=48;
geo.lin=@(x,xn,u) eval_lin_restr(cf,x,xn,u);
geo.bound_eval=@(x) eval_bounds_restr(cf,x,numel(idx_bnd));
fprintf('Derivadas das restricoes: 20 igualdades e %d desigualdades por estagio.\n',numel(g));
end

function [Jh,h0,Jg,g0]=eval_lin_restr(cf,x,xn,u)
GG=pack_pair_restr(x,xn,u);
Jh=full(cf.Jh(GG));
h0=full(cf.h(GG)); h0=h0(:);
Jg=full(cf.Jg(GG));
g0=full(cf.g(GG)); g0=g0(:);
end

function [gb,Jgb]=eval_bounds_restr(cf,x,nb)
GG=pack_pair_restr(x,x,zeros(8,1));
g=full(cf.g(GG));
J=full(cf.Jg(GG));
gb=g(1:2*nb); Jgb=J(1:2*nb,1:20);
end

function GG=pack_pair_restr(x,xn,u)
G1=pack_group_restr(x,u); G2=pack_group_restr(xn,zeros(8,1));
GG=[G1(:);G2(:)];
end

function G=pack_group_restr(x,u)
assert(numel(x)==30 && numel(u)==8);
G=eye(33);
G(1:3,1:3)=reshape(x(14:22),3,3);
for j=0:3
    b=23+2*j; ii=4+2*j;
    c=x(b); s=x(b+1);
    G(ii:ii+1,ii:ii+1)=[c,s;-s,c];
end
G(12:24,33)=x(1:13);
G(25:32,33)=u(:);
end

function x=unpack_group_state_restr(G)
x=[G(12:24,33); reshape(G(1:3,1:3),9,1); ...
   G(4,4);G(4,5);G(6,6);G(6,7); ...
   G(8,8);G(8,9);G(10,10);G(10,11)];
end

function S=lie_algebra_33_restr(w)
import casadi.*
S=SX.zeros(33,33);
v=w(14:16);
S(1:3,1:3)=[0,-v(3),v(2);v(3),0,-v(1);-v(2),v(1),0];
for j=0:3
    a=w(17+j); ii=4+2*j;
    S(ii:ii+1,ii:ii+1)=[0,a;-a,0];
end
S(12:24,33)=w(1:13);
S(25:32,33)=w(21:28);
end
