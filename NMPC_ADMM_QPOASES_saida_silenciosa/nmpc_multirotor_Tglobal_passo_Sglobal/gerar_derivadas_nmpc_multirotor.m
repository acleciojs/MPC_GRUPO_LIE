function prob = gerar_derivadas_nmpc_multirotor(cfg)
% GERAR_DERIVADAS_NMPC_MULTIROTOR — T_GLOBAL E S_GLOBAL
%
% Abordagem do MPC do usuario: UMA matriz simbolica T_global, UMA algebra
% S_global e T_sym={T_global} para obter H, grad f, Jg e Jh em grupo produto.
% Original do multirrotor: estados 30, incrementos intrinsecos 20, controles 8.
%
% Cada no e representado por G_k de 33x33:
%  - subgrupo translacional R^21: 13 coordenadas euclidianas + 8 controles;
%  - SO(3): 9 entradas de matriz com 3 incrementos;
%  - SO(2)^4: 4 pares [cos;sin] com 4 incrementos.
% T_global = blkdiag(G_0,...,G_N) esparsa, com N+1 nos.
% S_global = blkdiag(S_0,...,S_N) com omega_global em R^(20*(N+1)+8*N).
% O U terminal nao e variavel: o bloco final tem 8 controles fixados em zero.
%
% Hessiana do objetivo, Jacobianas de igualdade/desigualdade e gradiente
% sao construidos chamando as rotinas pedidas UMA vez para o grupo GLOBAL:
%   symbolic_Hessian_simetrica(f_T,{T_global},S_global,omega_global,0)
%   symbolic_Jacobian(f_T,{T_global},S_global,omega_global)
%   symbolic_Jacobian(g_T,{T_global},S_global,omega_global)
%   symbolic_Jacobian(h_T,{T_global},S_global,omega_global)
%
% T_global tem sparsity esparsa em blocos para nao alocar 33^2*(N+1)^2
% variaveis simbolicas que sao estruturalmente zero.
% expm_casadi usa jato de 2a ordem APENAS para derivadas em delta=0;
% passo_exponencial_sqp avalia S_global(Delta) e aplica expm numerica exata.
% Usa o modelo/custo originais fornecidos pelo usuario.
% Requer Init, Equilibrio, IntegrationEuler, Attitude_error e CasADi.
import casadi.*
if nargin<1 || isempty(cfg), cfg=struct(); end
N=read_cfg(cfg,'N',30); M=read_cfg(cfg,'M',15);
dt=read_cfg(cfg,'dt',0.01);
assert(N>=1 && M>=1 && mod(N,M)==0 && N==floor(N) && M==floor(M), ...
    'N e M inteiros, com N multiplo de M');
na=30; nt=20; nu=8;
nd=nt*(N+1)+nu*N;
np=na*(N+1)+nu*N;

Q=zeros(18,18);
Q(1,1)=50;Q(2,2)=25;Q(3,3)=50;
Q(4,4)=40;Q(5,5)=15;Q(6,6)=30;

Q(7:9,7:9)=20*eye(3);
Q(10:13,10:13)=20*eye(4);
Q(14:18,14:18)=10*eye(5);
R=diag([0.001,0.001,0.001,0.001,0.05,0.05,0.05,0.05]);
k1=read_cfg(cfg,'k1',10); k2=read_cfg(cfg,'k2',10);
if isfield(cfg,'Q'),Q=cfg.Q;end
if isfield(cfg,'R'),R=cfg.R;end
assert(isequal(size(Q),[18 18]) && isequal(size(R),[8 8]), ...
    'Q deve ser 18x18 e R deve ser 8x8');
u_min=read_cfg(cfg,'u_min',[0;0;0;0;-2;-2;-2;-2]);
u_max=read_cfg(cfg,'u_max',[80;80;80;80;2;2;2;2]);
u_min=u_min(:);u_max=u_max(:);
assert(numel(u_min)==8 && all(u_min<u_max),'Limites de controle invalidos');
ixb=(1:13).';
x_min=[-50;-50;0;-20;-20;-20;-20;-20;-20;-20;-20;-20;-20];
x_max=[50;50;50;20;20;20;20;20;20;10;10;10;10];
if isfield(cfg,'x_min'),x_min=cfg.x_min(:);end
if isfield(cfg,'x_max'),x_max=cfg.x_max(:);end
assert(numel(x_min)==13 && numel(x_max)==13 && all(x_min<x_max), ...
    'Limites de estado invalidos');

p_ref=read_cfg(cfg,'p_ref',[15;15;15]);p_ref=p_ref(:);
p_init=read_cfg(cfg,'p_init',[1;1;1]);p_init=p_init(:);
assert(numel(p_ref)==3 && numel(p_init)==3,'p_ref e p_init devem ter 3 elementos');
xi=zeros(30,1);xi(1:3)=p_ref;
xref=full(Init(xi));xref=xref(:);
xi=zeros(30,1);xi(1:3)=p_init;
xinit=full(Init(xi));xinit=xinit(:);
assert(numel(xref)==30 && numel(xinit)==30,'Init deve produzir 30 estados');
u_hover=full(Equilibrio(xref));u_hover=u_hover(:);
assert(numel(u_hover)==8 && all(isfinite(u_hover)), ...
    'Equilibrio deve produzir oito entradas finitas');
assert(all(u_hover>=u_min) && all(u_hover<=u_max), ...
    'Controle de equilibrio fora dos limites fornecidos');

%% ===== Grupo PRODUTO GLOBAL: T_sym={T_global}, S_global =====
K=N+1; nr=33; nG=nr*K;
X=SX.sym('X_base_NMPC',np,1);
d=SX.sym('delta_NMPC',nd,1);
omega_global=SX.sym('omega_global',nd,1);

% A matriz global tem suporte esparso em blocos 33x33. O suporte contem
% diagonais do grupo de translacoes, suas translacoes, SO(3), SO(2)^4.
% SX.zeros(33,33) em CasADi cria matriz densa: USAR Sparsity(33,33).
Gpat=SX(Sparsity(nr,nr));
Gpat(1:22,1:22)=SX.eye(22);
Gpat(1:21,22)=SX.sym('pat_translation',21,1);
Gpat(23:25,23:25)=SX.sym('pat_SO3',3,3);
for j=1:4
    rr=26+2*(j-1)+(0:1);
    Gpat(rr,rr)=SX.sym(sprintf('pat_SO2_%d',j),2,2);
end
patcell=repmat({Gpat},1,K);
Tpat=diagcat(patcell{:});
T_global=SX.sym('T_global',Tpat.sparsity());
T_sym={T_global};

S_cell=cell(K,1);
for k=1:K
    ix=(k-1)*nt+(1:nt);
    if k<=N
        iu=nt*K+(k-1)*nu+(1:nu);
        domega_u=omega_global(iu);
    else
        domega_u=SX.zeros(nu,1);  % SEM controle terminal ficticio
    end
    S_cell{k}=algebra_no_33(omega_global(ix),domega_u);
end
S_global=diagcat(S_cell{:});
assert(numel(T_sym)==1 && size(T_global,1)==nG, ...
    'O grupo global deve conter UMA matriz simbolica');
assert(numel(omega_global)==nd && size(S_global,1)==nG, ...
    'Dimensoes da algebra global incorretas');

% Condicao: construir TODO o custo e as restricoes DIRETAMENTE a partir de
% submatrizes da UNICA matriz T_global, nao de G_list={G0,...,GN}.
T_node=cell(K,1);
for k=1:K
    rr=(k-1)*nr+(1:nr);
    T_node{k}=T_global(rr,rr);
end

fT=SX(0);
gT=SX.zeros(0,1);
hT=SX.zeros(0,1);
for k=1:N
    [xk,uk]=extrair_no_33(T_node{k});
    [xn,~]=extrair_no_33(T_node{k+1});
    ef=[xk(1:13)-xref(1:13); ...
        Attitude_error(xk(14:22),xref(14:22),k1,k2); ...
        Attitude_error(xk(23:24),xref(23:24),k1,k2); ...
        Attitude_error(xk(25:26),xref(25:26),k1,k2); ...
        Attitude_error(xk(27:28),xref(27:28),k1,k2); ...
        Attitude_error(xk(29:30),xref(29:30),k1,k2)];
    fT=fT+ef.'*Q*ef+(uk-u_hover).'*R*(uk-u_hover);

    % 20 igualdades intrinsecas de dinamica por intervalo:
    % 13 euclidianas + 3 SO3 + 4 SO2.
    pred=IntegrationEuler(xk,uk,dt);pred=pred(:);
    assert(numel(pred)==na,'IntegrationEuler deve retornar 30 estados');
    E=reshape(pred(14:22),3,3).'*reshape(xn(14:22),3,3);
    eR=0.5*[E(3,2)-E(2,3); E(1,3)-E(3,1); E(2,1)-E(1,2)];
    e2=SX.zeros(4,1);
    for j=1:4
        jj=23+2*(j-1);
        e2(j)=pred(jj)*xn(jj+1)-pred(jj+1)*xn(jj);
    end
    gT=[gT; xn(1:13)-pred(1:13); eR; e2];

    % 26 limites de estado em x_(k+1), 16 de controle em u_k.
    hT=[hT; xn(ixb)-x_max; x_min-xn(ixb); ...
         uk-u_max;u_min-uk];
end
assert(numel(gT)==20*N && numel(hT)==42*N, ...
    'Quantidade de restricoes incorreta');

fprintf(['Derivadas no grupo PRODUTO GLOBAL: T_sym={T_global}, ' ...
    'S_global=%dx%d, delta=%d; N=%d ...\n'],nG,nG,nd,N);

% =============================================================
% CHAMADAS EXATAS PEDIDAS: somente UMA matriz T_global em T_sym.
% =============================================================
H_T = symbolic_Hessian_simetrica(fT,T_sym,S_global,omega_global,0);
grad_T = symbolic_Jacobian(fT,T_sym,S_global,omega_global).';
Jg_T = symbolic_Jacobian(gT,T_sym,S_global,omega_global);
Jh_T = symbolic_Jacobian(hT,T_sym,S_global,omega_global);
assert(isequal(size(H_T),[nd nd]) && isequal(size(Jg_T),[20*N nd]) ...
    && isequal(size(Jh_T),[42*N nd]),'Dimensoes globais invalidas');

% Construir a matriz BASE global usando exatamente os mesmos componentes
% fisicos das variaveis de entrada X. No ultimo no controle=0 e FIXO.
Gbase_cell=cell(K,1);
for k=1:K
    ia=(k-1)*na+(1:na);
    if k<=N
        iu=na*K+(k-1)*nu+(1:nu);
        ub=X(iu);
    else
        ub=SX.zeros(nu,1);
    end
    Gbase_cell{k}=montar_no_33(X(ia),ub);
end
Tbase=diagcat(Gbase_cell{:});
Tbase=project(Tbase,T_global.sparsity());
sb=Tbase.sparsity(); st=T_global.sparsity();
assert(sb.is_equal(st), ...
    'Sparsity de Tbase difere da T_global');

% Expressao do pullback para f,g,h: Tbase exp(S_global(delta)).
% O jato quadratico de exp garante derivadas de ordens 1 e 2 em delta=0.
Sdelta=substitute(S_global,omega_global,d);
Tpull=project(Tbase*expm_casadi(Sdelta),T_global.sparsity());
f_sym=substitute(fT,T_global,Tpull);
g_sym=substitute(gT,T_global,Tpull);
h_sym=substitute(hT,T_global,Tpull);

% As derivadas H_T,grad_T,Jg_T,Jh_T ja estao avaliadas em omega=0 e
% dependem somente da base. Substituir UNA VEZ T_global por Tbase.
H_sym=substitute(H_T,T_global,Tbase);
H_sym=0.5*(H_sym+H_sym.');
grad_sym=substitute(grad_T,T_global,Tbase);
Jg_sym=substitute(Jg_T,T_global,Tbase);
Jh_sym=substitute(Jh_T,T_global,Tbase);

% Funcao CasADi reutilizavel em cada SQP sem refazer o grafo.
fn=Function('multirotor_Tglobal_derivadas',{X,d}, ...
    {f_sym,grad_sym,H_sym,g_sym,Jg_sym,h_sym,Jh_sym});

% Passo exponencial: usar EXATAMENTE o mesmo S_global das derivadas.
% Funcao CasADi avalia o gerador S_global(Delta) no incremento ADMM.
% A base Tbase(X) e a mesma matriz do grupo produto usada no pullback.
% Ambas devolvem matrizes ESTRUTURALMENTE ESPARSAS de 33*(N+1).
% No passo numerico, explora-se a estrutura diagonal por blocos:
%   T_new,k = Tbase_k * expm(S_global,k(Delta)), k=0,...,N.
% Isto e exatamente igual a Tbase*expm(S_global), sem calcular
% a exponencial densa de uma matriz 1353x1353 quando N=40.
exp_cfg=struct();
exp_cfg.mode='T_global_S_global';
exp_cfg.T_global_fun=Function('T_global_base_retracao',{X},{Tbase});
exp_cfg.S_global_fun=Function('S_global_passo_retracao',{omega_global},{S_global});
exp_cfg.node_size=nr;
exp_cfg.num_nodes=K;
exp_cfg.n_state=na;
exp_cfg.n_state_tangent=nt;
exp_cfg.n_control=nu;
exp_cfg.n_delta=nd;
exp_cfg.n_base=np;

% Blocos temporais sem condensacao: 20*(bs+1)+8*bs variaveis por bloco.
indices_blocos=cell(M,1);bs=N/M;
for i=1:M
    k0=(i-1)*bs;
    ids=zeros(nt*(bs+1)+nu*bs,1);pos=0;
    for j=0:bs
        ids(pos+(1:nt))=(k0+j)*nt+(1:nt);pos=pos+nt;
    end
    for j=0:bs-1
        ids(pos+(1:nu))=nt*K+(k0+j)*nu+(1:nu);pos=pos+nu;
    end
    indices_blocos{i}=ids;
end

prob=struct();
prob.f_sym=f_sym;prob.grad_sym=grad_sym;prob.H_sym=H_sym;
prob.g_sym=g_sym;prob.Jg_sym=Jg_sym;
prob.h_sym=h_sym;prob.Jh_sym=Jh_sym;
prob.delta_sym=d;prob.X_sym=X;
prob.fn=fn;prob.indices_blocos=indices_blocos;
prob.exp_cfg=exp_cfg;
prob.N=N;prob.M=M;prob.bs=bs;prob.dt=dt;
prob.na=na;prob.nt=nt;prob.nu=nu;prob.nd=nd;prob.np=np;
prob.n_group=nG;prob.d_global=nd;
prob.T_sym=T_sym;prob.T_global=T_global;prob.S_global=S_global;
prob.omega_global=omega_global;
prob.derivative_backend='T_global + S_global (uma matriz grupo produto)';
prob.xref=xref;prob.xinit=xinit;prob.u_hover=u_hover;
prob.Q=Q;prob.R=R;prob.k1=k1;prob.k2=k2;
prob.x_min=x_min;prob.x_max=x_max;prob.u_min=u_min;prob.u_max=u_max;
xx=SX.sym('x_model',30,1);uu=SX.sym('u_model',8,1);
prob.model_fun=Function('modelo_discreto_multirotor',{xx,uu}, ...
    {IntegrationEuler(xx,uu,dt)});
fprintf('T_global/S_global: grupo=%dx%d | delta=%d | H=%dx%d | Jg=%dx%d | Jh=%dx%d\n', ...
    nG,nG,nd,size(H_sym,1),size(H_sym,2),size(Jg_sym,1),size(Jg_sym,2), ...
    size(Jh_sym,1),size(Jh_sym,2));
end

function S=algebra_no_33(dx,du)
% Gerador aditivo R^21 + so3 + so2^4. MATRIZ ESTRUTURALMENTE ESPARSA.
import casadi.*
S=SX(Sparsity(33,33));
S(1:21,22)=[dx(1:13);du(:)];
v=dx(14:16);
S(23:25,23:25)=[0,-v(3),v(2); v(3),0,-v(1);-v(2),v(1),0];
for j=1:4
    rr=26+2*(j-1)+(0:1);
    a=dx(16+j);
    S(rr,rr)=[0,-a;a,0];
end
end

function G=montar_no_33(x,u)
% Imersao de um estado ambiente (30) e um controle (8) na matriz 33x33.
import casadi.*
G=SX(Sparsity(33,33));
G(1:22,1:22)=SX.eye(22);
G(1:21,22)=[x(1:13);u(:)];
G(23:25,23:25)=reshape(x(14:22),3,3);
for j=1:4
    ii=23+2*(j-1); rr=26+2*(j-1)+(0:1);
    c=x(ii);s=x(ii+1);
    G(rr,rr)=[c,-s;s,c];
end
end

function [x,u]=extrair_no_33(G)
% Mesmas convencoes do estado original multirrotor: 13 + 9 + 8 = 30.
x=[G(1:13,22);reshape(G(23:25,23:25),9,1)];
for j=1:4
    ii=26+2*(j-1);
    x=[x;G(ii,ii);G(ii+1,ii)]; %#ok<AGROW>
end
u=G(14:21,22);
end

function v=read_cfg(cfg,field,def)
if isfield(cfg,field) && ~isempty(cfg.(field)),v=cfg.(field);else,v=def;end
end
