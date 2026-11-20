function testar_passo_exponencial_Sglobal()
% TESTAR_PASSO_EXPONENCIAL_SGLOBAL
% Teste sem Init/Equilibrio/IntegrationEuler; requer CasADi.
% Verifica a identidade numérica:
%   Tplus=Tbase*expm(S_global(delta))
% e que o desempacotamento executado pela SQP é o mesmo.
import casadi.*
rng(12);
K=3; N=K-1; nr=33; na=30; nt=20; nu=8;
np=na*K+nu*N;
nd=nt*K+nu*N;
X=SX.sym('test_X_base',np,1);
omega=SX.sym('test_omega_global',nd,1);
Tc=cell(K,1); Sc=cell(K,1);
for k=1:K
    xi=(k-1)*na+(1:na);
    di=(k-1)*nt+(1:nt);
    if k<=N
        ui=na*K+(k-1)*nu+(1:nu);
        du=omega(nt*K+(k-1)*nu+(1:nu));
        uk=X(ui);
    else
        du=SX.zeros(nu,1);
        uk=SX.zeros(nu,1);
    end
    Tc{k}=pack33(X(xi),uk);
    Sc{k}=lie33(omega(di),du);
end
Tbase=diagcat(Tc{:});
Sglobal=diagcat(Sc{:});
cfg=struct();
cfg.mode='T_global_S_global';
cfg.T_global_fun=Function('test_T_global_base',{X},{Tbase});
cfg.S_global_fun=Function('test_S_global_eval',{omega},{Sglobal});
cfg.num_nodes=K;
cfg.node_size=nr;
cfg.n_state=na;
cfg.n_state_tangent=nt;
cfg.n_control=nu;
cfg.n_delta=nd;
cfg.n_base=np;

x=zeros(np,1);
for k=1:K
    i=(k-1)*na;
    x(i+(1:13))=randn(13,1);
    axis=randn(3,1);
    R=expm(skew3(axis));
    x(i+(14:22))=R(:);
    for j=1:4
        th=0.2*randn();
        ii=i+23+2*(j-1);
        x(ii:ii+1)=[cos(th);sin(th)];
    end
end
x(na*K+(1:nu*N))=randn(nu*N,1);
d=0.1*randn(nd,1);

% Referência por uma UNICA exponencial global pequena (99x99).
T=full(cfg.T_global_fun(x));
S=full(cfg.S_global_fun(d));
Tnext=T*expm(S);
xref=zeros(np,1);
for k=1:K
    rr=(k-1)*nr+(1:nr);
    Gn=Tnext(rr,rr);
    ix=(k-1)*na+(1:na);
    xref(ix(1:13))=Gn(1:13,22);
    xref(ix(14:22))=reshape(Gn(23:25,23:25),9,1);
    for j=1:4
        ir=26+2*(j-1);
        id=23+2*(j-1);
        xref(ix(id:id+1))=[Gn(ir,ir);Gn(ir+1,ir)];
    end
    if k<=N
        iu=na*K+(k-1)*nu+(1:nu);
        xref(iu)=Gn(14:21,22);
    end
end

xout=passo_exponencial_sqp(x,d,cfg);
err=norm(xout-xref,inf);
ident=norm(passo_exponencial_sqp(x,zeros(nd,1),cfg)-x,inf);

fprintf('Passo S_GLOBAL: erro vs expm global = %.3e; identidade = %.3e\n',err,ident);
assert(err<2e-10 && ident<2e-10, ...
    'A atualizacao SQP nao coincide com T_global*expm(S_global(delta))');
fprintf('TESTE PASSO EXPONENCIAL S_GLOBAL PASSOU.\n');
end

function G=pack33(x,u)
import casadi.*
G=SX(Sparsity(33,33));
G(1:22,1:22)=SX.eye(22);
G(1:21,22)=[x(1:13);u];
G(23:25,23:25)=reshape(x(14:22),3,3);
for j=1:4
    id=23+2*(j-1);
    rr=26+2*(j-1)+(0:1);
    c=x(id);s=x(id+1);
    G(rr,rr)=[c -s;s c];
end
end

function S=lie33(dx,du)
import casadi.*
S=SX(Sparsity(33,33));
S(1:21,22)=[dx(1:13);du];
a=dx(14:16);
S(23:25,23:25)=skew3(a);
for j=1:4
    rr=26+2*(j-1)+(0:1);
    th=dx(16+j);
    S(rr,rr)=[0,-th;th,0];
end
end

function K=skew3(v)
K=[0,-v(3),v(2);v(3),0,-v(1);-v(2),v(1),0];
end
