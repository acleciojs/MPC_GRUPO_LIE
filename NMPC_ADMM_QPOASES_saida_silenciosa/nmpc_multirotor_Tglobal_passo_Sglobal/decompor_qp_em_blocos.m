function blocos = decompor_qp_em_blocos(H,f,g0,Jg,h0,Jh,indices,lb,ub,opts)
% DECOMPOR_QP_EM_BLOCOS
% Decomposicao EXATA da QP (antes da regularizacao local):
%  min 1/2*d'*H*d + f'*d
%  s.a. Jg*d=-g0; Jh*d<=-h0; lb<=d<=ub.
% Qualquer acoplamento de Hessiana ou restricao nao coberto por um bloco
% causa erro explicito. Uma coordenada repetida deve ocupar blocos
% consecutivos, para o consenso adjacente garantir igualdade global.

n=numel(f); M=numel(indices);
H=sparse(0.5*(H+H.'));f=f(:);
assert(isequal(size(H),[n n]),'Hessiana mal dimensionada');
assert(size(Jg,2)==n && size(Jh,2)==n,'Jacobianas mal dimensionadas');
assert(numel(g0)==size(Jg,1) && numel(h0)==size(Jh,1), ...
       'Dimensao dos residuos nao corresponde as Jacobianas');
assert(numel(lb)==n && numel(ub)==n,'Limites mal dimensionados');
indices=cellfun(@(v)v(:),indices,'UniformOutput',false);
inc=false(n,M);
for i=1:M
    id=indices{i};
    assert(all(id>=1 & id<=n & id==round(id)) && ...
        numel(unique(id))==numel(id),'Indices de bloco invalidos');
    inc(id,i)=true;
end
assert(all(any(inc,2)),'Alguma coordenada de delta nao pertence a bloco');
for j=1:n
    occ=find(inc(j,:));
    assert(all(diff(occ)==1), ...
       'A coordenada global %d reaparece em blocos nao consecutivos',j);
end

% Distribuir cada monomio da forma quadratica para EXATAMENTE um bloco.
Hloc=cell(M,1);floc=cell(M,1);
for i=1:M
    Hloc{i}=sparse(numel(indices{i}),numel(indices{i}));
    floc{i}=zeros(numel(indices{i}),1);
end
for j=1:n
    owner=find(inc(j,:),1,'last');
    pos=find(indices{owner}==j,1);
    floc{owner}(pos)=f(j);
end
[r,c,v]=find(triu(H));
for q=1:numel(v)
    owner=find(inc(r(q),:) & inc(c(q),:),1,'last');
    assert(~isempty(owner), ...
       ['H(%d,%d) = %.3e acopla variaveis sem um bloco comum. ' ...
        'Aumente o bloco/overlap ou reformule o consenso.'],r(q),c(q),v(q));
    ir=find(indices{owner}==r(q),1);
    ic=find(indices{owner}==c(q),1);
    Hloc{owner}(ir,ic)=Hloc{owner}(ir,ic)+v(q);
    if ir~=ic
        Hloc{owner}(ic,ir)=Hloc{owner}(ic,ir)+v(q);
    end
end

% As linhas sao atribuidas ao primeiro bloco que contem todo seu suporte.
EQrows=cell(M,1); INrows=cell(M,1);
for i=1:M, EQrows{i}=[];INrows{i}=[];end
for row=1:size(Jg,1)
    support=find(Jg(row,:)~=0);
    if isempty(support)
        assert(abs(g0(row))<=1e-12, ...
            'Igualdade constante nao satisfaz g(%d)=0',row);
        continue;
    end
    owner=find(all(inc(support,:),1),1,'first');
    assert(~isempty(owner), ...
       'Linha de igualdade %d acopla blocos sem suporte conjunto',row);
    EQrows{owner}(end+1)=row; %#ok<AGROW>
end
for row=1:size(Jh,1)
    support=find(Jh(row,:)~=0);
    if isempty(support)
        assert(h0(row)<=1e-12, ...
            'Desigualdade constante inviavel na linha %d',row);
        continue;
    end
    owner=find(all(inc(support,:),1),1,'first');
    assert(~isempty(owner), ...
       'Linha de desigualdade %d acopla blocos sem suporte conjunto',row);
    INrows{owner}(end+1)=row; %#ok<AGROW>
end

blocos=cell(M,1);
for i=1:M
    id=indices{i};
    m=numel(id);
    Hi=full(0.5*(Hloc{i}+Hloc{i}.'));
    % UNICA regularizacao: deslocamento diagonal positivo por bloco.
    eigmin=min(eig(Hi));
    reg=max(opts.reg_base,opts.reg_floor-eigmin);
    Hi=Hi+reg*eye(m);
    b=struct();
    b.id=id;b.n=m;b.H=0.5*(Hi+Hi.');b.f=floc{i};
    b.Aeq=full(Jg(EQrows{i},id));b.beq=-g0(EQrows{i});
    b.Aineq=full(Jh(INrows{i},id));b.bineq=-h0(INrows{i});
    b.lb=lb(id);b.ub=ub(id);
    b.reg_delta=reg;
    b.Ls=sparse(0,m);b.Le=sparse(0,m);
    b.interface_start=[];b.interface_end=[];
    blocos{i}=b;
end

for i=1:M-1
    shared=intersect(indices{i},indices{i+1});
    assert(~isempty(shared), ...
      'Blocos %d e %d nao compartilham variaveis; sem interface',i,i+1);
    [~,posE]=ismember(shared,indices{i});
    [~,posS]=ismember(shared,indices{i+1});
    k=numel(shared);
    blocos{i}.Le=sparse(1:k,posE,1,k,blocos{i}.n);
    blocos{i+1}.Ls=sparse(1:k,posS,1,k,blocos{i+1}.n);
    blocos{i}.interface_end=shared;
    blocos{i+1}.interface_start=shared;
end
end
