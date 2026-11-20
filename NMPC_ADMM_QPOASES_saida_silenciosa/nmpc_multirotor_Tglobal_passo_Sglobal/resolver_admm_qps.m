function [delta,info] = resolver_admm_qps(blocos,opts)
% RESOLVER_ADMM_QPS  ADMM ESCALAR OU PRECONDICIONADO POR INTERFACE.
% EXTERNA a sqp_lie_admm.m
%
% Cada bloco tem w_i, H_i, f_i, Aeq_i, Aineq_i, lb_i, ub_i.
% Interfaces: Le_i*w_i = z_i = Ls_(i+1)*w_(i+1).
% Multiplicadores gammae_i, gammas_(i+1) sao NAO escalados.
% rho fixo, sem Nesterov. P_i = rho*diag(p_i), com p_i>0.
% Warm start opcional: opts.admm_seed (w, z, gamma_e, gamma_s).
% As variaveis de consenso e os multiplicadores sao NAO escalados.
% As QPs locais usam qpOASES/ProxQP via CasADi conic, sem quadprog.
% Reutilizam solucoes primal-duais e a instancia por bloco.
% O MESMO P_i e usado nas duas copias da interface i.
% Modos opts.precond_mode: 'scalar', 'curvature', 'manual'.
% Em 'curvature', os pesos sao estimados pela diagonal de Hessianas
% RESTRITAS AOS INDICES DE INTERFACE (aproximacao, nao Schur exato),
% normalizados e limitados para robustez.
% Em 'manual', opts.precond_weights e um vetor por interface ou cell.
% Sem aceitacao inexata: exige r<=eps_pr, s<=eps_du e gap<=tol_gap.
% A QP central nao e alterada: precondicionamento atua APENAS no consenso.

M=numel(blocos);nvar=zeros(M,1);
for i=1:M,nvar(i)=blocos{i}.n;end
rho=opts.rho;
precond_mode=lower(char(field_or(opts,'precond_mode','scalar')));
assert(ismember(precond_mode,{'scalar','curvature','manual'}), ...
    'precond_mode deve ser scalar, curvature ou manual');
max_iter=opts.max_iter;
tol_abs=opts.tol_abs;tol_rel=opts.tol_rel;
tol_gap=field_or(opts,'tol_gap_inf',1e-5);
% Tolerancias NUMERICAS do QP local, independentes do consenso ADMM.
% Nao confundir com tol_ineq_lin: essa ultima valida o passo SQP global.
qp_constr_tol=field_or(opts,'qp_constraint_tolerance',1e-5);
qp_opt_tol=field_or(opts,'qp_optimality_tolerance',1e-8);
qp_max_iter=field_or(opts,'qp_max_iterations',800);
qp_local_eq_tol=field_or(opts,'qp_local_eq_tol',1e-6);
qp_local_ineq_tol=field_or(opts,'qp_local_ineq_tol',1e-5);
assert(all(isfinite([qp_constr_tol,qp_opt_tol,qp_local_eq_tol,qp_local_ineq_tol])) ...
    && all([qp_constr_tol,qp_opt_tol,qp_local_eq_tol,qp_local_ineq_tol]>0), ...
    'Tolerancias locais da QP devem ser finitas e positivas');
assert(isfinite(qp_max_iter) && qp_max_iter>=1 && qp_max_iter==floor(qp_max_iter), ...
    'qp_max_iterations deve ser inteiro positivo');
assert(isfinite(tol_gap) && tol_gap>0,'tol_gap_inf deve ser positivo');
assert(max_iter==round(max_iter) && max_iter>=1,'max_iter invalido');
assert(rho>0 && tol_abs>0 && tol_rel>=0,'rho/tol invalidos');

niface=M-1;
[metric_diag,metric_weights]=construir_metricas(blocos,rho,precond_mode,opts);
z=cell(niface,1);gamma_e=cell(niface,1);gamma_s=cell(niface,1);
for i=1:niface
    k=size(blocos{i}.Le,1);
    assert(k==size(blocos{i+1}.Ls,1),'Interface %d inconsistente',i);
    z{i}=zeros(k,1);gamma_e{i}=zeros(k,1);gamma_s{i}=zeros(k,1);
end
w=cell(M,1);
for i=1:M,w{i}=zeros(nvar(i),1);end

% Warm start externo. Cada copia de interface recebe duais anti-simetricos.
% A seed vem recentrada na NOVA trajetoria nominal (nao reaplica delta antigo).
seed=field_or(opts,'admm_seed',[]);
[ws, zs, ges, gss, warm_used, warm_reason]=validar_seed(seed,blocos);
if warm_used
    w=ws;z=zs;gamma_e=ges;gamma_s=gss;
end
warm_start_qp=logical(field_or(opts,'admm_warm_local_qp',true));
warm_qp_accepted=0;
warm_qp_rejected=0;

% H augmented nao muda durante esta QP: construir UMA VEZ para
% evitar recalcular produtos matriciais a cada iteracao ADMM.
Haug=cell(M,1);
% Uma instancia qpOASES por bloco permanece viva durante TODO o ADMM.
% A estrutura da QP local nao muda: apenas ci varia com z/gamma.
Haug=cell(M,1);
qp_solver=cell(M,1);
qp_lam_a=cell(M,1);qp_lam_x=cell(M,1);
solver_name=lower(char(field_or(opts,'solver','qpoases')));
assert(ismember(solver_name,{'qpoases','proxqp'}), ...
       'Solver permitido: qpoases ou proxqp (CasADi conic); sem quadprog');
for i=1:M
    b=blocos{i};Hi=b.H;
    if i>1
        pvec=metric_diag{i-1};L=b.Ls;
        Hi=Hi+L.'*(spdiags(pvec,0,numel(pvec),numel(pvec))*L);
    end
    if i<M
        pvec=metric_diag{i};L=b.Le;
        Hi=Hi+L.'*(spdiags(pvec,0,numel(pvec),numel(pvec))*L);
    end
    Haug{i}=sparse(0.5*(Hi+Hi.'));
    qp_solver{i}=qp_casadi_criar(Haug{i},b.Aeq,b.beq, ...
        b.Aineq,b.bineq,b.lb,b.ub,solver_name, ...
        qp_max_iter,sprintf('qp_admm_bloco_%d',i));
    qp_lam_a{i}=zeros(qp_solver{i}.m,1);
    qp_lam_x{i}=zeros(b.n,1);
end

info=struct('converged',false,'ok',true,'reason','max_iter', ...
    'iter',0,'r_pr',Inf,'r_du',Inf,'eps_pr',Inf,'eps_du',Inf, ...
    'gap_inf',Inf,'gap_threshold',tol_gap,'history',nan(max_iter,4), ...
    'precond_mode',precond_mode,'metric_weights',{metric_weights}, ...
    'metric_diag',{metric_diag}, ...
    'z',{z},'gamma_e',{gamma_e},'gamma_s',{gamma_s},'w',{w}, ...
    'warm_used',warm_used,'warm_reason',warm_reason, ...
    'warm_qp_accepted',0,'warm_qp_rejected',0, ...
    'qp_constraint_tolerance',qp_constr_tol, ...
    'qp_local_eq_tol',qp_local_eq_tol,'qp_local_ineq_tol',qp_local_ineq_tol, ...
    'max_local_eq_violation',0,'max_local_ineq_violation',0, ...
    'qp_solver',solver_name,'qp_calls',0,'qp_restarts',0, ...
    'qp_iter_total',0, ...
    'warm_dual_inf',max_norm_cell(gamma_e), ...
    'warm_primal_inf',max_norm_cell(w));

for iter=1:max_iter
    zold=z;
    % P1: resolver as QPs locais com z e duais CONGELADOS.
    for i=1:M
        b=blocos{i};Hi=Haug{i};ci=b.f;
        if i>1
            L=b.Ls;pvec=metric_diag{i-1};
            ci=ci+L.'*(gamma_s{i-1}-pvec.*z{i-1});
        end
        if i<M
            L=b.Le;pvec=metric_diag{i};
            ci=ci+L.'*(gamma_e{i}-pvec.*z{i});
        end
        try
            % Reutilizar o resultado PRIMAL/DUAL da iteracao anterior.
            % A instancia conic ja conserva a estrutura da QP e possibilita
            % hotstart interno do qpOASES quando o plugin permitir.
            x0=w{i};
            canwarm=warm_start_qp && viavel_qp_local(x0,b, ...
                max(qp_local_eq_tol,qp_local_ineq_tol));
            if canwarm
                warm_qp_accepted=warm_qp_accepted+1;
                xseed=x0;
                la0=qp_lam_a{i};lx0=qp_lam_x{i};
            else
                if warm_start_qp, warm_qp_rejected=warm_qp_rejected+1; end
                xseed=zeros(b.n,1);
                la0=zeros(qp_solver{i}.m,1);lx0=zeros(b.n,1);
            end
            try
                [wi,lama,lamx,st]=qp_casadi_resolver( ...
                    qp_solver{i},ci,xseed,la0,lx0);
                verificar_resolucao_local(wi,b,qp_local_eq_tol,qp_local_ineq_tol);
            catch MEoriginal
                % Uma falha de hotstart nao deve necessariamente parar o MPC:
                % reiniciar SOMENTE o solver local que falhou e tentar novamente.
                % Nao aceitamos solucoes apenas pela viabilidade: o plugin
                % deve indicar convergencia.
                info.qp_restarts=info.qp_restarts+1;
                if logical(field_or(opts,'qp_restart_on_failure',true))
                    try
                        qp_solver{i}=qp_casadi_criar(Hi,b.Aeq,b.beq, ...
                            b.Aineq,b.bineq,b.lb,b.ub,solver_name, ...
                            max(2*qp_max_iter,qp_max_iter+100), ...
                            sprintf('qp_admm_reset_b%d_it%d',i,iter));
                        [wi,lama,lamx,st]=qp_casadi_resolver( ...
                            qp_solver{i},ci,zeros(b.n,1), ...
                            zeros(qp_solver{i}.m,1),zeros(b.n,1));
                        verificar_resolucao_local(wi,b, ...
                            qp_local_eq_tol,qp_local_ineq_tol);
                    catch MEretry
                        error(['QP local falhou (original): %s; ' ...
                            'reinicio: %s'],MEoriginal.message,MEretry.message);
                    end
                else
                    rethrow(MEoriginal);
                end
            end
            qp_lam_a{i}=lama;qp_lam_x{i}=lamx;
            info.qp_calls=info.qp_calls+1;
            if isfield(st,'iter_count') && isnumeric(st.iter_count)
                info.qp_iter_total=info.qp_iter_total+double(st.iter_count);
            end
            assert(all(isfinite(wi)) && numel(wi)==b.n, ...
                'QP local retornou valores invalidos');
            eqr=norm(b.Aeq*wi-b.beq,inf);
            ineqr=max([0;b.Aineq*wi-b.bineq;b.lb-wi;wi-b.ub]);
            info.max_local_eq_violation=max(info.max_local_eq_violation,eqr);
            info.max_local_ineq_violation=max(info.max_local_ineq_violation,ineqr);
            if eqr>qp_local_eq_tol*max(1,norm(b.beq,inf)) || ...
                    ineqr>qp_local_ineq_tol
                error(['QP local nao viavel: eq=%.3e (lim %.3e), ' ...
                    'ineq=%.3e (lim %.3e)'],eqr, ...
                    qp_local_eq_tol*max(1,norm(b.beq,inf)), ...
                    ineqr,qp_local_ineq_tol);
            end
            w{i}=wi(:);
        catch ME
            info.converged=false;info.ok=false;
            info.reason=sprintf('iter=%d, bloco=%d: %s',iter,i,ME.message);
            info.iter=iter;info.history=info.history(1:max(0,iter-1),:);
            info.w=w;info.z=z;info.gamma_e=gamma_e;info.gamma_s=gamma_s;
            info.warm_qp_accepted=warm_qp_accepted;
            info.warm_qp_rejected=warm_qp_rejected;
            delta=[];return;
        end
    end

    % P2: consenso corrigido pelos multiplicadores nao escalados.
    for i=1:niface
        xe=blocos{i}.Le*w{i};
        xs=blocos{i+1}.Ls*w{i+1};
        z{i}=0.5*(xe+xs)+0.5*((gamma_e{i}+gamma_s{i})./metric_diag{i});
    end
    % P3: multiplicadores para as duas copias de cada interface.
    for i=1:niface
        gamma_e{i}=gamma_e{i}+metric_diag{i}.*(blocos{i}.Le*w{i}-z{i});
        gamma_s{i}=gamma_s{i}+metric_diag{i}.*(blocos{i+1}.Ls*w{i+1}-z{i});
    end

    r2=0;s2=0;Ax2=0;Bz2=0;y2=0;
    gap_inf=0; ncopy=0;
    % Norma do residual dual usa A'*(rho*B*Delta z), nao apenas ||Dz||.
    for i=1:niface
        xe=blocos{i}.Le*w{i};
        xs=blocos{i+1}.Ls*w{i+1};
        re=xe-z{i};rs=xs-z{i};
        dz=z{i}-zold{i};
        r2=r2+sum(re.^2)+sum(rs.^2);
        Ax2=Ax2+sum(xe.^2)+sum(xs.^2);
        Bz2=Bz2+2*sum(z{i}.^2);
        pdz=metric_diag{i}.*dz;
        s2=s2+norm(blocos{i}.Le.'*pdz)^2 + ...
                 norm(blocos{i+1}.Ls.'*pdz)^2;
        gap_inf=max(gap_inf,max(abs(xe-xs)));
        ncopy=ncopy+2*numel(z{i});
    end
    for i=1:M
        y=zeros(blocos{i}.n,1);
        if i>1, y=y+blocos{i}.Ls.'*gamma_s{i-1};end
        if i<M, y=y+blocos{i}.Le.'*gamma_e{i};end
        y2=y2+norm(y)^2;
    end
    r_pr=sqrt(r2);r_du=sqrt(s2);
    eps_pr=sqrt(ncopy)*tol_abs+tol_rel*max(sqrt(Ax2),sqrt(Bz2));
    eps_du=sqrt(sum(nvar))*tol_abs+tol_rel*sqrt(y2);
    info.iter=iter;info.r_pr=r_pr;info.r_du=r_du;
    info.eps_pr=eps_pr;info.eps_du=eps_du;
    info.gap_inf=gap_inf;
    info.history(iter,:)=[r_pr,r_du,eps_pr,eps_du];
    if opts.verbose && (iter==1 || mod(iter,opts.verbose_every)==0)
        fprintf(' ADMM[%s]=%d | r/eps=%.3e | s/eps=%.3e | gap=%.3e (lim %.1e)\n', ...
                precond_mode,iter,r_pr/max(eps_pr,eps),r_du/max(eps_du,eps),gap_inf,tol_gap);
    end
    if r_pr<=eps_pr && r_du<=eps_du && gap_inf<=tol_gap
        info.converged=true;info.reason='converged';
        break;
    end
end
info.history=info.history(1:info.iter,:);
info.z=z;info.gamma_e=gamma_e;info.gamma_s=gamma_s;
info.w=w;
info.warm_qp_accepted=warm_qp_accepted;
info.warm_qp_rejected=warm_qp_rejected;
% Recuperacao por media das copias locais: consistente no limite primal.
n=max(cellfun(@(b)max(b.id),blocos));
sums=zeros(n,1);counts=zeros(n,1);
for i=1:M
    id=blocos{i}.id;
    sums(id)=sums(id)+w{i};
    counts(id)=counts(id)+1;
end
delta=sums./counts;
end



function [w,z,ge,gs,ok,reason]=validar_seed(seed,blocos)
% Aceita apenas seeds com dimensoes identicas e entradas finitas.
% Preserva gamma NAO escalado; P nao deve multiplicar gamma na mudanca de QP.
M=numel(blocos);ns=M-1;
w=cell(M,1);z=cell(ns,1);ge=cell(ns,1);gs=cell(ns,1);
for i=1:M,w{i}=zeros(blocos{i}.n,1);end
for i=1:ns
    d=size(blocos{i}.Le,1);
    z{i}=zeros(d,1);ge{i}=zeros(d,1);gs{i}=zeros(d,1);
end
ok=false;reason='cold_start';
if isempty(seed) || ~isstruct(seed),return;end
fields={'w','z','gamma_e','gamma_s'};
for j=1:numel(fields)
    if ~isfield(seed,fields{j}) || ~iscell(seed.(fields{j}))
        reason=['seed_sem_' fields{j}];return;
    end
end
if numel(seed.w)~=M || numel(seed.z)~=ns || ...
    numel(seed.gamma_e)~=ns || numel(seed.gamma_s)~=ns
    reason='seed_dimensoes_blocos';return;
end
for i=1:M
    v=seed.w{i}(:);
    if numel(v)~=blocos{i}.n || ~all(isfinite(v))
        reason='seed_w_invalido';return;
    end
end
for i=1:ns
    d=size(blocos{i}.Le,1);
    if numel(seed.z{i})~=d || numel(seed.gamma_e{i})~=d || ...
            numel(seed.gamma_s{i})~=d || ...
            ~all(isfinite(seed.z{i}(:))) || ...
            ~all(isfinite(seed.gamma_e{i}(:))) || ...
            ~all(isfinite(seed.gamma_s{i}(:)))
        reason='seed_interface_invalida';return;
    end
end
for i=1:M,w{i}=seed.w{i}(:);end
for i=1:ns
    z{i}=seed.z{i}(:);
    % Em qualquer otimo de consenso, gamma_e+gamma_s=0.
    % A projecao evita carregamento de drift numérico/seed inconsistente.
    lambda=0.5*(seed.gamma_e{i}(:)-seed.gamma_s{i}(:));
    ge{i}=lambda;gs{i}=-lambda;
end
ok=true;reason='seed_valida';
if isfield(seed,'source') && ischar(seed.source)
    reason=['seed_valida_' seed.source];
end
end

function yes=viavel_qp_local(x,b,tol)
% Optimizacao warm x0 so quando a QP local possui a MESMA base de restricoes
% ou o chute transportado satisfaz as restricoes linearizadas atuais.
yes=false;
if isempty(x) || numel(x)~=b.n || any(~isfinite(x)),return;end
x=x(:);
if any(x<b.lb-tol) || any(x>b.ub+tol),return;end
if ~isempty(b.Aeq)
    if norm(b.Aeq*x-b.beq,inf)>tol*(1+norm(b.beq,inf)),return;end
end
if ~isempty(b.Aineq)
    if max(b.Aineq*x-b.bineq)>tol,return;end
end
yes=true;
end

function v=max_norm_cell(c)
v=0;
for j=1:numel(c)
    if ~isempty(c{j}),v=max(v,norm(c{j},inf));end
end
end

function [Pdiag,weights]=construir_metricas(blocos,rho,mode,opts)
% P_i = rho*diag(weights_i) com pesos positivos e FIXOS por QP.
% Os pesos usados em ambos os lados de uma interface sao identicos.
M=numel(blocos);ns=M-1;
Pdiag=cell(ns,1);weights=cell(ns,1);
clamp=field_or(opts,'precond_clamp',[0.2 5]);
assert(isnumeric(clamp) && numel(clamp)==2 && all(isfinite(clamp)) && ...
       clamp(1)>0 && clamp(2)>=clamp(1), ...
       'precond_clamp deve ser [min,max] com min>0');
power=field_or(opts,'precond_power',0.5);
assert(isscalar(power) && isfinite(power) && power>=0 && power<=1, ...
       'precond_power precisa estar em [0,1]');
for i=1:ns
    Le=blocos{i}.Le;Ls=blocos{i+1}.Ls;
    d=size(Le,1);
    assert(d>0 && size(Ls,1)==d,'Interface %d invalida',i);
    if strcmp(mode,'scalar')
        wi=ones(d,1);
    elseif strcmp(mode,'manual')
        W=field_or(opts,'precond_weights',[]);
        if iscell(W)
            assert(numel(W)==ns,'precond_weights: cell com %d interfaces',ns);
            wi=W{i}(:);
        elseif isvector(W) && ~isempty(W)
            wi=W(:);
        else
            error('precond_weights deve ser vetor positivo ou cell de vetores');
        end
        assert(numel(wi)==d && all(isfinite(wi)) && all(wi>0), ...
               'Pesos manuais invalidos na interface %d',i);
    else
        % Curvatura diagonal nos estados compartilhados (nao eh o Schur).
        He=blocos{i}.H; Hs=blocos{i+1}.H;
        ce=full(diag(Le*He*Le.'));
        cs=full(diag(Ls*Hs*Ls.'));
        assert(numel(ce)==d && numel(cs)==d, 'Diagonal nao encontrada');
        floor_curv=1e-12;
        ce=max(ce,floor_curv);cs=max(cs,floor_curv);
        raw=sqrt(ce.*cs);
        base=median(raw);
        wi=(raw/max(base,floor_curv)).^power;
        wi=min(clamp(2),max(clamp(1),wi));
    end
    Pdiag{i}=rho*wi;
    weights{i}=wi;
end
end

function verificar_resolucao_local(wi,b,eq_tol,ineq_tol)
assert(numel(wi)==b.n && all(isfinite(wi)), ...
    'Solver CasADi retornou valores invalidos');
eqr=norm(b.Aeq*wi-b.beq,inf);
ine= max([0;b.Aineq*wi-b.bineq;b.lb-wi;wi-b.ub]);
if eqr > eq_tol*max(1,norm(b.beq,inf)) || ine > ineq_tol
    error('QP local nao viavel: eq=%.3e ineq=%.3e',eqr,ine);
end
end

function v=field_or(s,key,def)
if isfield(s,key) && ~isempty(s.(key)),v=s.(key);else,v=def;end
end
