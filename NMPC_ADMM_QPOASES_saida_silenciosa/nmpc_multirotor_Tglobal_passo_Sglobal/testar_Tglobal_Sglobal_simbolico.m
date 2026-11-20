function testar_Tglobal_Sglobal_simbolico()
% TESTE INDEPENDENTE: verifica T_sym={T_global}, S_global e derivadas
% sem depender do modelo do multirrotor. Requer apenas CasADi e os dois
% arquivos symbolic_Hessian_simetrica.m e symbolic_Jacobian.m.
import casadi.*
n=33;K=2;nt=20;nu=8;nd=nt*K+nu;

B=SX(Sparsity(n,n));
B(1:22,1:22)=SX.eye(22);
B(1:21,22)=SX.sym('p',21);
B(23:25,23:25)=SX.sym('R',3,3);
for j=1:4
    rr=26+2*(j-1)+(0:1);
    B(rr,rr)=SX.sym(sprintf('Q%d',j),2,2);
end
Gpattern=diagcat(B,B);
T_global=SX.sym('T_global_test',Gpattern.sparsity());
T_sym={T_global};

omega_global=SX.sym('omega_global_test',nd,1);
S1=algebra_test(omega_global(1:20),omega_global(41:48));
S2=algebra_test(omega_global(21:40),SX.zeros(8,1));
S_global=diagcat(S1,S2);
G0=T_global(1:n,1:n);
G1=T_global(n+(1:n),n+(1:n));

% Custos dependem de ambos os estados, de controles e rotacoes.
fT=sumsqr(G0(1:3,22)-[1;2;3])+0.1*sumsqr(G0(14:21,22)) ...
  +sumsqr(G0(23:25,23)-[1;0;0]) ...
  +0.2*sumsqr(G1(1:3,22)) ...
  +0.5*G0(26,26)*G1(27,26);
% Restricao entre dois nos e controle.
gT=G1(1:3,22)-G0(1:3,22)-G0(14:16,22);
hT=[G0(14:21,22)-ones(8,1);-G0(14:21,22)];

H_T=symbolic_Hessian_simetrica(fT,T_sym,S_global,omega_global,0);
G_T=symbolic_Jacobian(fT,T_sym,S_global,omega_global).';
Jg_T=symbolic_Jacobian(gT,T_sym,S_global,omega_global);
Jh_T=symbolic_Jacobian(hT,T_sym,S_global,omega_global);

% Bases validas do grupo produto: identidade de SO3 e SO2.
X=SX.sym('Xtest',30*K+nu,1);
Tbase=diagcat(pack_test(X(1:30),X(61:68)), ...
              pack_test(X(31:60),SX.zeros(8,1)));
Tbase=project(Tbase,T_global.sparsity());
d=SX.sym('dtest',nd,1);
Tpull=Tbase*expm_casadi(substitute(S_global,omega_global,d));
Tpull=project(Tpull,T_global.sparsity());
f=substitute(fT,T_global,Tpull);
g=substitute(gT,T_global,Tpull);
h=substitute(hT,T_global,Tpull);
H=substitute(H_T,T_global,Tbase);
G=substitute(G_T,T_global,Tbase);
Jg=substitute(Jg_T,T_global,Tbase);
Jh=substitute(Jh_T,T_global,Tbase);
Hd=substitute(hessian(f,d),d,zeros(nd,1));
Gd=substitute(jacobian(f,d).',d,zeros(nd,1));
Jgd=substitute(jacobian(g,d),d,zeros(nd,1));
Jhd=substitute(jacobian(h,d),d,zeros(nd,1));

checker=Function('testar_global_jac_hess',{X}, ...
    {H-Hd,G-Gd,Jg-Jgd,Jh-Jhd});
x0=zeros(30*K+nu,1);
for k=1:K
    i=(k-1)*30;
    x0(i+(14:22))=reshape(eye(3),9,1);
    x0(i+[23,25,27,29])=1;
end
[a,b,c,e]=checker(x0);
err=max([norm(full(a),inf),norm(full(b),inf), ...
    norm(full(c),inf),norm(full(e),inf)]);
fprintf('T_GLOBAL/S_GLOBAL | dimensao=%dx%d | d=%d | erro derivadas = %.3e\n', ...
    n*K,n*K,nd,err);
assert(err<1e-9,'Derivadas globais inconsistentes');
fprintf('TESTE T_GLOBAL PASSOU.\n');
end

function S=algebra_test(dx,du)
import casadi.*
S=SX(Sparsity(33,33));
S(1:21,22)=[dx(1:13);du(:)];
v=dx(14:16);
S(23:25,23:25)=[0,-v(3),v(2);v(3),0,-v(1);-v(2),v(1),0];
for j=1:4
    rr=26+2*(j-1)+(0:1);a=dx(16+j);
    S(rr,rr)=[0,-a;a,0];
end
end

function G=pack_test(x,u)
import casadi.*
G=SX(Sparsity(33,33));
G(1:22,1:22)=SX.eye(22);
G(1:21,22)=[x(1:13);u(:)];
G(23:25,23:25)=reshape(x(14:22),3,3);
for j=1:4
    idx=23+2*(j-1);rr=26+2*(j-1)+(0:1);
    c=x(idx);s=x(idx+1);
    G(rr,rr)=[c,-s;s,c];
end
end
