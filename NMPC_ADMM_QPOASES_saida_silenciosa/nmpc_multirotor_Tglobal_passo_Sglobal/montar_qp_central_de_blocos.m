function qp = montar_qp_central_de_blocos(blocos,n)
% MONTA QP CENTRAL DE REFERENCIA PARA A MESMA QP REGULARIZADA DOS BLOCOS.
% Variaveis de interface sao identificadas pelo mesmo indice global.
% Esta funcao NAO e usada para resolver a SQP: somente comparacao numerica.
H=sparse(n,n);f=zeros(n,1);
neq=sum(cellfun(@(b)numel(b.beq),blocos));
niq=sum(cellfun(@(b)numel(b.bineq),blocos));
Aeq=sparse(neq,n);Aineq=sparse(niq,n);
beq=zeros(neq,1);bineq=zeros(niq,1);
lb=-inf(n,1);ub=inf(n,1);
re=0;ri=0;
for i=1:numel(blocos)
    b=blocos{i};ix=b.id;
    H(ix,ix)=H(ix,ix)+sparse(b.H);
    f(ix)=f(ix)+b.f;
    ne=numel(b.beq);nh=numel(b.bineq);
    if ne>0
        Aeq(re+(1:ne),ix)=sparse(b.Aeq);
        beq(re+(1:ne))=b.beq;
    end
    if nh>0
        Aineq(ri+(1:nh),ix)=sparse(b.Aineq);
        bineq(ri+(1:nh))=b.bineq;
    end
    lb(ix)=max(lb(ix),b.lb);
    ub(ix)=min(ub(ix),b.ub);
    re=re+ne;ri=ri+nh;
end
qp=struct('H',0.5*(H+H.'),'f',f,'Aeq',Aeq,'beq',beq, ...
    'Aineq',Aineq,'bineq',bineq,'lb',lb,'ub',ub);
end
