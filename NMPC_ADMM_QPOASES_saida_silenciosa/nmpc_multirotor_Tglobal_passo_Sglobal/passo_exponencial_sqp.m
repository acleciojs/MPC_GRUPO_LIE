function xnew = passo_exponencial_sqp(x,delta,exp_cfg)
% PASSO_EXPONENCIAL_SQP: modo T_GLOBAL/S_GLOBAL exato ou legado por bases.
%
% exp_cfg.euclid_x_idx: indices ambientes das coordenadas Euclidianas.
% exp_cfg.euclid_d_idx: indices tangentes correspondentes.
%
% exp_cfg.groups(j).x_idx: indices ambientes de vec(X_j), POR COLUNAS.
% exp_cfg.groups(j).delta_idx: indices das coordenadas tangentes de X_j.
% exp_cfg.groups(j).basis: matriz r x r x d contendo a base S_j.
%
% Realiza X_j <- X_j*expm(sum_l delta_l*basis(:,:,l)).
% Para SO(2) armazenado em par [cos(theta);sin(theta)], use
% groups(j).representation='so2-pair', x_idx com 2 valores e
% basis(:,:,1)=[0 -1;1 0]; atualiza q <- expm(delta*B)*q.
% Nenhuma projecao nem reducao de passo. alpha_SQP=1.

% Quando exp_cfg veio do gerador T_GLOBAL/S_GLOBAL, NAO reconstruir
% geradores por bases independentes. Avaliar o proprio S_global(Delta).
if isfield(exp_cfg,'mode') && strcmpi(exp_cfg.mode,'T_global_S_global')
    xnew=passo_Tglobal_Sglobal_exato(x,delta,exp_cfg);
    return;
end

x=x(:);delta=delta(:);xnew=x;
used=zeros(numel(delta),1);
if isfield(exp_cfg,'euclid_x_idx') && ~isempty(exp_cfg.euclid_x_idx)
    ix=exp_cfg.euclid_x_idx(:);
    id=exp_cfg.euclid_d_idx(:);
    assert(numel(ix)==numel(id),'Indices euclidianos incompatíveis');
    assert(all(id>=1 & id<=numel(delta)) && ...
           all(ix>=1 & ix<=numel(x)), ...
           'Indice euclidiano fora das dimensoes');
    used(id)=used(id)+1;
    xnew(ix)=xnew(ix)+delta(id);
end
if ~isfield(exp_cfg,'groups') || isempty(exp_cfg.groups)
    assert(all(used==1), 'Coordenadas tangentes sem mapeamento ou duplicadas');
    return;
end
for j=1:numel(exp_cfg.groups)
    q=exp_cfg.groups(j);
    ix=q.x_idx(:);id=q.delta_idx(:);B=q.basis;
    r=size(B,1);
    assert(size(B,2)==r && size(B,3)==numel(id), ...
        'Base da algebra mal dimensionada no grupo %d',j);
    repr='matriz';
    if isfield(q,'representation') && ~isempty(q.representation)
        repr=char(q.representation);
    end
    assert(numel(ix)==r*r || (strcmpi(repr,'so2-pair') && r==2 && numel(ix)==2), ...
        'Indices ambientes invalidos no grupo %d',j);
    assert(all(id>=1 & id<=numel(delta)) && ...
           all(ix>=1 & ix<=numel(x)), ...
           'Indice de grupo fora das dimensoes');
    used(id)=used(id)+1;
    if strcmpi(repr,'so2-pair')
        % O MPC armazena SO(2) como [cos(theta);sin(theta)], nao vec(R).
        assert(numel(id)==1 && numel(ix)==2 && r==2, ...
            'SO2 pair requer uma coordenada angular e dois indices ambientes');
        xnew(ix)=expm(delta(id)*B(:,:,1))*x(ix);
        continue;
    end
    X=reshape(x(ix),r,r);
    Lie=zeros(r,r);
    for ell=1:numel(id)
        Lie=Lie+delta(id(ell))*B(:,:,ell);
    end
    Xnext=X*expm(Lie);
    xnew(ix)=Xnext(:);
end
assert(all(used==1), 'Coordenadas tangentes sem mapeamento ou duplicadas');
assert(all(isfinite(xnew)),'Passo exponencial produziu NaN/Inf');
end


function xnew=passo_Tglobal_Sglobal_exato(x,delta,cfg)
% PASSO_TGLOBAL_SGLOBAL_EXATO
% Uma unica fonte da algebra: S_global_fun foi compilada a partir
% da MESMA expressao S_global usada na Hessiana e Jacobianas.
%
% T_global^+ = T_global * expm(S_global(delta))
% Como ambas as matrizes sao diagonais em blocos de 33x33, a exponencial
% e calculada EXATAMENTE por bloco, evitando expm de 1353x1353.
%
% No ultimo no, S_global nao possui incremento de controle; o controle
% terminal foi fixado em zero na propria construcao simbolica.

x=x(:);delta=delta(:);
K=cfg.num_nodes;
nr=cfg.node_size;
na=cfg.n_state;
nu=cfg.n_control;
nt=cfg.n_state_tangent;
N=K-1;
assert(nr==33 && na==30 && nt==20 && nu==8, ...
    'Configuracao T_GLOBAL/S_GLOBAL incompatível com o multirrotor');
assert(numel(x)==cfg.n_base && numel(delta)==cfg.n_delta, ...
    'Dimensoes da base ou do incremento nao coincidem com o gerador');
assert(all(isfinite(x)) && all(isfinite(delta)), ...
    'T_GLOBAL/S_GLOBAL recebeu dados nao finitos');

% CasADi DM com esparsidade bloco-diagonal; NAO converter a matriz
% global completa para double, pois ela pode ter 1353x1353 elementos.
TDM=cfg.T_global_fun(x);
SDM=cfg.S_global_fun(delta);
assert(isequal(size(TDM),[nr*K nr*K]) && ...
       isequal(size(SDM),[nr*K nr*K]), ...
       'T_global ou S_global com dimensao incorreta');

xnew=zeros(size(x));
for k=1:K
    rr=(k-1)*nr+(1:nr);
    Tk=full(TDM(rr,rr));
    Sk=full(SDM(rr,rr));

    % E = exp(S_global,k(delta)) NUMERICA EXATA.
    % Matematicamente: diagcat(exp(S1),...,exp(SK)) = exp(S_global).
    Tnext=Tk*expm(Sk);

    ix=(k-1)*na+(1:na);
    xnew(ix(1:13))=Tnext(1:13,22);
    xnew(ix(14:22))=reshape(Tnext(23:25,23:25),9,1);
    for j=1:4
        ir=26+2*(j-1);
        ii=23+2*(j-1);
        xnew(ix(ii:ii+1))=[Tnext(ir,ir);Tnext(ir+1,ir)];
    end
    if k<=N
        iu=na*K+(k-1)*nu+(1:nu);
        xnew(iu)=Tnext(14:21,22);
    end
end
assert(all(isfinite(xnew)),'Passo T_GLOBAL/S_GLOBAL produziu NaN ou Inf');
end
