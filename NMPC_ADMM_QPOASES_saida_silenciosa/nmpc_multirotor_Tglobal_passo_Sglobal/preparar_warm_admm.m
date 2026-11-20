function seed=preparar_warm_admm(info,Xbase_old,Xbase_new, ...
                    Xtarget,Ubase_new,Utarget,indices_blocos,bs,shift,opts)
% PREPARAR_WARM_ADMM  Warm start primal-dual geometricamente recentrado.
%
% O cache anterior contém multiplicadores NAO ESCALADOS do consenso.
% Se shift=1, interpola os duais nas interfaces temporais uma amostra a
% frente; se shift=0, apenas transporta do ponto-base SQP antigo ao novo.
%
% IMPORTANTE: os w antigos NÃO sao reaplicados! O vetor primal inicial e
% log_{Xbase_new}(Xtarget) nas coordenadas S_GLOBAL, que e zero se a
% nova nominal ja coincide com a trajetoria otimizada anterior.
% Os duais SO(3) sao transportados entre bases de perturbacao a direita.
% Nos SO(2), o transporte e identidade. O restante e euclidiano.
%
% Entradas:
% Xbase_old  30 x (N+1), ponto de linearizacao da QP antiga
% Xbase_new  30 x (N+1), novo ponto de linearizacao
% Xtarget    30 x (N+1), alvo fisico no novo horizonte
% Ubase_new  8 x N, controles nominais novos
% Utarget    8 x N, controles alvo
% indices_blocos indices globais das variaveis locais (com duplicatas)
% bs         numero de intervalos por bloco
% shift      0: mesma linha temporal (nova SQP), 1: nova amostra MPC
%
% Opcional opts.admm_warm_dual_damping em [0,1], padrao 0.95;
% opts.admm_warm_primal_damping em [0,1], padrao 1;
% opts.admm_warm_primal_max_inf>0, padrao 0.5.

M=numel(indices_blocos);ns=M-1;
N=size(Ubase_new,2);nt=20;nu=8;K=N+1;
assert(size(Xbase_old,1)==30 && isequal(size(Xbase_old),[30,K]) && ...
       isequal(size(Xbase_new),[30,K]) && isequal(size(Xtarget),[30,K]), ...
       'Estados do warm ADMM devem ser 30 x (N+1)');
assert(isequal(size(Ubase_new),[nu,N]) && isequal(size(Utarget),[nu,N]), ...
       'Controles do warm ADMM devem ser 8 x N');
assert(M>=2 && bs==round(bs) && bs>0 && bs*M==N, ...
       'N/M incompatível para shift das interfaces');
assert(shift==0 || shift==1,'shift do ADMM deve ser 0 ou 1');
assert(isfield(info,'converged') && info.converged, ...
       'Nao fazer warm start de ADMM nao convergido');
assert(isfield(info,'gamma_e') && numel(info.gamma_e)==ns && ...
       isfield(info,'gamma_s') && numel(info.gamma_s)==ns, ...
       'Numero anterior de interfaces nao corresponde ao atual');

damping=getopt(opts,'admm_warm_dual_damping',0.95);
pdamping=getopt(opts,'admm_warm_primal_damping',1.0);
maxprimal=getopt(opts,'admm_warm_primal_max_inf',0.5);
assert(isfinite(damping) && damping>=0 && damping<=1, ...
       'admm_warm_dual_damping fora de [0,1]');
assert(isfinite(pdamping) && pdamping>=0 && pdamping<=1, ...
       'admm_warm_primal_damping fora de [0,1]');
assert(maxprimal>0 && isfinite(maxprimal), ...
       'admm_warm_primal_max_inf deve ser finito e positivo');

% Primal recenter: exp(S(d)) aplicado em Xbase_new reproduz Xtarget,
% salvo erros de arredondamento e o limitador conservador.
Dx=zeros(nt,K);
for k=1:K
    Dx(1:13,k)=Xtarget(1:13,k)-Xbase_new(1:13,k);
    R0=reshape(Xbase_new(14:22,k),3,3);
    Rt=reshape(Xtarget(14:22,k),3,3);
    Dx(14:16,k)=log_so3(R0.'*Rt);
    for j=1:4
        idx=23+2*(j-1);
        c0=Xbase_new(idx,k);s0=Xbase_new(idx+1,k);
        ct=Xtarget(idx,k);st=Xtarget(idx+1,k);
        Dx(16+j,k)=atan2(c0*st-s0*ct,c0*ct+s0*st);
    end
end
Du=Utarget-Ubase_new;
delta=[Dx(:);Du(:)];
assert(all(isfinite(delta)),'Warm primal geometricamente invalido');
mx=max(abs(delta));
if mx>maxprimal
    delta=delta*(maxprimal/mx);
end
delta=pdamping*delta;
seed=struct();seed.w=cell(M,1);
for i=1:M
    ids=indices_blocos{i}(:);
    seed.w{i}=delta(ids);
end
seed.z=cell(ns,1);seed.gamma_e=cell(ns,1);seed.gamma_s=cell(ns,1);
for i=1:ns
    row_i=i*bs+1;
    zi=delta((row_i-1)*nt+(1:nt));
    seed.z{i}=zi(:);

    if shift==0 || i==ns
        % No ultimo intervalo falta o proximo multiplicador antigo:
        % persistir o ultimo valor (com damping global).
        lam=transportar_lam(info,i,Xbase_old,Xbase_new,i,bs);
    else
        % Um passo no horizonte antigo = 1/bs do bloco seguinte.
        alpha=1/bs;
        la=transportar_lam(info,i,Xbase_old,Xbase_new,i,bs);
        lb=transportar_lam(info,i+1,Xbase_old,Xbase_new,i,bs);
        lam=(1-alpha)*la+alpha*lb;
    end
    lam=damping*lam;
    seed.gamma_e{i}=lam;
    seed.gamma_s{i}=-lam;
end
seed.source='sqp_recenter';
if shift==1,seed.source='mpc_shift';end
seed.primal_inf=norm(delta,inf);
seed.dual_inf=max(cellfun(@(c)norm(c,inf),seed.gamma_e));
seed.shift=shift;
end

function lam=transportar_lam(info,j,Xold,Xnew,i,bs)
% Coordenadas tangentes de SO(3) sao perturbacoes a direita.
% Rold * exp(hat(vold)) e Rnew * exp(hat(vnew)):
% vnew = Rnew' * Rold * vold, na linearizacao. O mesmo para covetores
% sob identificacao euclidiana, pois a transformacao e ortogonal.
oldnode=j*bs+1;
newnode=i*bs+1;
le=info.gamma_e{j}(:);
ls=info.gamma_s{j}(:);
assert(numel(le)==20 && numel(ls)==20 && ...
       all(isfinite([le;ls])),'Dual antigo com dimensão invalida');
lam=0.5*(le-ls);
Ro=reshape(Xold(14:22,oldnode),3,3);
Rn=reshape(Xnew(14:22,newnode),3,3);
lam(14:16)=Rn.'*Ro*lam(14:16);
end

function v=log_so3(R)
% Logaritmo principal de uma rotacao SO(3) para ||theta||<pi.
q=(trace(R)-1)/2;q=min(1,max(-1,q));theta=acos(q);
a=[R(3,2)-R(2,3);R(1,3)-R(3,1);R(2,1)-R(1,2)];
if theta<1e-6
    v=0.5*a;
elseif pi-theta<1e-5
    % Caso raro proximo de pi: rotina matricial com verificação.
    L=logm(R);
    if max(abs(imag(L(:))))>1e-6
        error('Warm start de SO3 proximo de pi ficou ambiguo');
    end
    L=real(L);
    v=0.5*[L(3,2)-L(2,3);L(1,3)-L(3,1);L(2,1)-L(1,2)];
else
    v=(theta/(2*sin(theta)))*a;
end
end

function v=getopt(s,name,def)
if isfield(s,name) && ~isempty(s.(name)),v=s.(name);else,v=def;end
end
