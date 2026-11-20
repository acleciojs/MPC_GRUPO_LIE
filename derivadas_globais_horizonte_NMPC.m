function obj = derivadas_globais_horizonte_NMPC(N,Q,Rc,k1,k2,modo,expm_mode)
% DERIVADAS_GLOBAIS_HORIZONTE_NMPC
% Hessiana geometrica de F = sum_{k=0}^{N-1} ell(x_k,u_k), sem custo terminal.
%
% Grupo de cada no: G_k in R^(33x33), com SO(3) x SO(2)^4 x R^13 x R^8.
% G_sym = {G_0, ..., G_N}; o controle ficticio do ultimo no e descartado.
% Coordenadas GLOBAIS de retorno:
%   omega_global = [dq_0; ...; dq_N; du_0; ...; du_(N-1)]
% Dimensao: 20*(N+1)+8*N.
%
% modo = 'global_direct': uma chamada a symbolic_hessian_simetrica_pullback
%                       para TODO o horizonte (custo simbolico alto).
%                       Pode requerer MUITA memoria para N=50.
% modo = 'sparse_equivalent': calcula a mesma Hessiana global por blocos
%                       simbolicos independentes de cada NO (F e separavel).
%                       Essa segunda opcao NAO faz chamada unica global.
%
% Somente Hessiana SIMETRICA do pullback exponencial, SEM A_skew.
% Deriva gradiente e Hessiana da MESMA funcao perturbada.
% expm_mode='jet2' usa exp(S) = I+S+S^2/2 + O(||S||^3):
%   os valores, gradientes e Hessianas em xi=0 sao EXATAMENTE os mesmos
%   da exponencial (para funcoes C2). Nao calcula expm_casadi.
% expm_mode='expm' usa expm_casadi(S) como no codigo original.
% Dependencias externas: CasADi, Attitude_error; opcionalmente expm_casadi.
% Nao inclui os arquivos proprietarios do modelo.
import casadi.*
assert(N>=1 && N==floor(N),'N deve ser inteiro positivo');
if nargin<6 || isempty(modo), modo='sparse_equivalent'; end
if nargin<7 || isempty(expm_mode), expm_mode='jet2'; end
nx=20; nu=8; d=28; ng=33;
nd=nx*(N+1)+nu*N;
param=SX.sym('ref_global',38,1);
xr=param(1:30);
ur=param(31:38);
w_local=SX.sym('omega_local',d,1);
S=lie_algebra_33_global(w_local);

switch lower(modo)
    case 'global_direct'
        fprintf(['GERANDO Hessiana SIMETRICA do objetivo GLOBAL: N=%d, ' ...
                 '%d nos de 33x33; %d coordenadas simbolicas; modo=%s.\n'], ...
                 N,N+1,d*(N+1),expm_mode);
        G_sym=cell(1,N+1);
        Gvec=SX.zeros(0,1);
        for k=0:N
            Gk=SX.sym(sprintf('G_global_%03d',k),ng,ng);
            G_sym{k+1}=Gk;
            Gvec=[Gvec;Gk(:)]; %#ok<AGROW>
        end
        F=SX(0);
        for k=1:N
            Gk=G_sym{k};
            xk=unpack_group_state_global(Gk);
            uk=Gk(25:32,33);
            F=F+cost_stage_global(xk,uk,xr,ur,Q,Rc,k1,k2);
        end
        % Esta e a chamada GLOBAL pedida: G_sym contem TODO o horizonte.
        [H_all,grad_all]=symbolic_hessian_simetrica_pullback( ...
            F,G_sym,S,w_local,expm_mode);
        assert(numel(grad_all)==d*(N+1), ...
            'Gradiente simetrico retornou ordem/dimensao inesperada');
        assert(isequal(size(H_all),[d*(N+1),d*(N+1)]), ...
            'Hessiana simetrica retornou dimensoes inesperadas');
        % Ordem original: [dq0;du0;dq1;du1;...;dqN;duN].
        % Ordem SQP: [dq0;...;dqN;du0;...;du(N-1)].
        ids_estado=zeros(1,nx*(N+1));
        for k=0:N
            ids_estado(k*nx+(1:nx))=k*d+(1:nx);
        end
        ids_controle=zeros(1,nu*N);
        for k=0:N-1
            ids_controle(k*nu+(1:nu))=k*d+nx+(1:nu);
        end
        keep=[ids_estado,ids_controle];
        H_global=H_all(keep,keep);
        g_global=grad_all(keep);
        cf.H=Function('H_objetivo_GLOBAL',{Gvec,param},{H_global});
        cf.g=Function('g_objetivo_GLOBAL',{Gvec,param},{g_global});
        cf.f=Function('F_objetivo_GLOBAL',{Gvec,param},{F});
        obj.eval=@(X,U,xref,uref) eval_global_direct(cf,X,U,xref,uref,N);

    case 'sparse_equivalent'
        fprintf(['GERANDO Hessiana GLOBAL ESPARSA equivalente: %d coordenadas; ' ...
                 'custo separavel por (x_k,u_k).\n'],nd);
        Gk=SX.sym('G_no',ng,ng);
        f=cost_stage_global(unpack_group_state_global(Gk),Gk(25:32,33), ...
                            xr,ur,Q,Rc,k1,k2);
        G_sym={Gk};
        [Hj,gj]=symbolic_hessian_simetrica_pullback( ...
            f,G_sym,S,w_local,expm_mode);
        assert(numel(gj)==d && isequal(size(Hj),[d,d]));
        cf.H=Function('H_no_GLOBAL',{Gk(:),param},{Hj});
        cf.g=Function('g_no_GLOBAL',{Gk(:),param},{gj});
        cf.f=Function('f_no_GLOBAL',{Gk(:),param},{f});
        obj.eval=@(X,U,xref,uref) eval_global_esparso(cf,X,U,xref,uref,N);

    otherwise
        error('Modo invalido: %s. Use global_direct ou sparse_equivalent.',modo);
end
obj.mode=modo;
obj.expm_mode=expm_mode;
obj.N=N;
obj.nq=nx;
obj.nu=nu;
obj.nd=nd;
obj.ordem='[dq_0;...;dq_N;du_0;...;du_(N-1)]';
end

function [H,g,f0]=eval_global_direct(cf,X,U,xr,ur,N)
assert(isequal(size(X),[30,N+1]) && isequal(size(U),[8,N]));
GG=zeros(33*33*(N+1),1);
for k=1:N+1
    if k<=N, uk=U(:,k); else, uk=zeros(8,1); end
    G=group_T_33_global(X(:,k),uk);
    GG((k-1)*1089+(1:1089))=G(:);
end
param=[xr(:);ur(:)];
H=sparse(full(cf.H(GG,param)));
g=full(cf.g(GG,param)); g=g(:);
f0=full(cf.f(GG,param));
check_derivatives(H,g,N);
end

function [H,g,f0]=eval_global_esparso(cf,X,U,xr,ur,N)
assert(isequal(size(X),[30,N+1]) && isequal(size(U),[8,N]));
nd=20*(N+1)+8*N;
H=spalloc(nd,nd,N*28*28);
g=zeros(nd,1); f0=0;
param=[xr(:);ur(:)];
for k=1:N
    G=group_T_33_global(X(:,k),U(:,k));
    Gv=G(:);
    Hk=full(cf.H(Gv,param));
    gk=full(cf.g(Gv,param));
    fk=full(cf.f(Gv,param));
    ids=[(k-1)*20+(1:20), 20*(N+1)+(k-1)*8+(1:8)];
    H(ids,ids)=H(ids,ids)+Hk;
    g(ids)=g(ids)+gk(:);
    f0=f0+fk;
end
check_derivatives(H,g,N);
end

function check_derivatives(H,g,N)
nd=20*(N+1)+8*N;
assert(isequal(size(H),[nd nd]) && numel(g)==nd, ...
       'Dimensoes erradas: Hessiana global/gradiente global.');
assert(all(isfinite(nonzeros(H))) && all(isfinite(g)), ...
       'Hessiana ou gradiente global contem NaN/Inf.');
end

function G=group_T_33_global(x,u)
assert(numel(x)==30 && numel(u)==8);
G=eye(33);
G(1:3,1:3)=reshape(x(14:22),3,3);
for j=0:3
    b=23+2*j; a=4+2*j;
    c=x(b); s=x(b+1);
    G(a:a+1,a:a+1)=[c,s;-s,c];
end
G(12:24,33)=x(1:13);
G(25:32,33)=u(:);
end

function x=unpack_group_state_global(G)
x=[G(12:24,33);reshape(G(1:3,1:3),9,1); ...
   G(4,4);G(4,5);G(6,6);G(6,7); ...
   G(8,8);G(8,9);G(10,10);G(10,11)];
end

function S=lie_algebra_33_global(omega)
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

function y=cost_stage_global(x,u,xr,ur,Q,Rc,k1,k2)
% Mesmo custo do controlador original; NAO ha custo terminal.
phiB=Attitude_error(x(14:22),xr(14:22),k1,k2);
phi1=Attitude_error(x(23:24),xr(23:24),k1,k2);
phi2=Attitude_error(x(25:26),xr(25:26),k1,k2);
phi3=Attitude_error(x(27:28),xr(27:28),k1,k2);
phi4=Attitude_error(x(29:30),xr(29:30),k1,k2);
e=[x(1:13)-xr(1:13);phiB;phi1;phi2;phi3;phi4];
y=e'*Q*e+(u-ur)'*Rc*(u-ur);
end
