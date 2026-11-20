function result=testar_warm_admm()
% TESTAR_WARM_ADMM  Testes independentes do VANT.
% Requer Optimization Toolbox (qpOASES). Testa:
% 1) mesma QP com cold e warm: mesmo otimo;
% 2) QP com gradiente perturbado: o warm nao altera o otimo;
% 3) transporte geometricamente consistente do dual em SO(3);
% 4) nao reaplicar incremento anterior na base otimizada.

opts=struct('rho',2,'max_iter',5000,'tol_abs',1e-9,...
    'tol_rel',1e-9,'tol_gap_inf',1e-8,'solver','qpoases',...
    'verbose',false,'verbose_every',200,'precond_mode','curvature',...
    'precond_power',0.5,'precond_clamp',[0.2 5]);
H1=diag([0.2 8]);H2=diag([30 0.1]);
b1=struct('n',2,'id',(1:2)','H',H1,'f',[-2;1],...
    'Aeq',[1 1],'beq',0.1,'Aineq',zeros(0,2),...
    'bineq',zeros(0,1),'lb',[-3;-3],'ub',[3;3],...
    'Le',speye(2),'Ls',sparse(0,2));
b2=struct('n',2,'id',(1:2)','H',H2,'f',[1;-1.5],...
    'Aeq',zeros(0,2),'beq',zeros(0,1),...
    'Aineq',[1 -1],'bineq',0.05,'lb',[-3;-3],'ub',[3;3],...
    'Le',sparse(0,2),'Ls',speye(2));
blocos={b1,b2};
[x1,a]=resolver_admm_qps(blocos,opts);
assert(a.converged && ~a.warm_used,'Primeiro solve deveria ser frio');
seed=struct('w',{a.w},'z',{a.z},'gamma_e',{a.gamma_e},...
    'gamma_s',{a.gamma_s},'source','qp_repetida');
opts.admm_seed=seed;
[x2,b]=resolver_admm_qps(blocos,opts);
assert(b.converged && b.warm_used && norm(x1-x2,inf)<2e-6,...
    'A mesma QP com warm start nao reproduziu a solucao fria');

% Perturbar suavemente custo linear e comparar com outro cold start.
blocos{1}.f=blocos{1}.f+[0.02;-0.015];
opts.admm_seed=[];
[xcold,ic]=resolver_admm_qps(blocos,opts);
opts.admm_seed=seed;
[xwarm,iw]=resolver_admm_qps(blocos,opts);
assert(ic.converged && iw.converged && ...
    norm(xcold-xwarm,inf)<2e-6, ...
    'Mudanca do objetivo: as solucoes cold/warm divergiram');

% Teste especifico do transporte geometrico SO3 e recenter do primal.
N=8;M=2;bs=4;nt=20;nu=8;
indices=cell(M,1);
for i=1:M
    ids=[];for k=(i-1)*bs:i*bs
        ids=[ids,k*nt+(1:nt)]; %#ok<AGROW>
    end
    for k=(i-1)*bs:i*bs-1
        ids=[ids,nt*(N+1)+k*nu+(1:nu)]; %#ok<AGROW>
    end
    indices{i}=ids(:);
end
Xold=zeros(30,N+1);Xnew=zeros(30,N+1);
for k=1:N+1
    Xold(14:22,k)=reshape(eye(3),9,1);
    Xnew(14:22,k)=reshape(eye(3),9,1);
    for j=1:4
        idx=23+2*(j-1);
        Xold(idx,k)=1;Xnew(idx,k)=1;
    end
end
phi=0.18;R=[cos(phi) -sin(phi) 0;sin(phi) cos(phi) 0;0 0 1];
Xnew(14:22,bs+1)=R(:);
Unew=zeros(nu,N);
info=struct('converged',true,...
  'gamma_e',{{[zeros(13,1);1;2;3;zeros(4,1)]}}, ...
  'gamma_s',{{-[zeros(13,1);1;2;3;zeros(4,1)]}});
wopt=struct('admm_warm_dual_damping',1.0,...
            'admm_warm_primal_damping',1.0,...
            'admm_warm_primal_max_inf',1.0);
s=preparar_warm_admm(info,Xold,Xnew,Xnew,Unew,Unew,indices,bs,1,wopt);
expected=R.'*[1;2;3];
assert(norm(s.gamma_e{1}(14:16)-expected,inf)<1e-10,...
    'Transporte SO3 de multiplicador incorreto');
assert(norm(s.gamma_e{1}+s.gamma_s{1},inf)<1e-12,...
    'Duplicatas de interface devem ter duais opostos');
assert(norm([s.w{1};s.w{2}],inf)<1e-12,...
    'Incremento antigo foi reaplicado na base otimizada');

result=struct('iter_1_cold',a.iter,'iter_1_warm',b.iter,...
    'iter_perturbado_cold',ic.iter,'iter_perturbado_warm',iw.iter,...
    'erro_QP',norm(xcold-xwarm,inf),'erro_rotacao',...
    norm(s.gamma_e{1}(14:16)-expected,inf));
fprintf(['WARM OK | repete: cold %d, warm %d | alterado: ' ...
    'cold %d, warm %d | erro QP %.2e | SO3 %.2e\n'],...
    a.iter,b.iter,ic.iter,iw.iter,result.erro_QP,result.erro_rotacao);
end
