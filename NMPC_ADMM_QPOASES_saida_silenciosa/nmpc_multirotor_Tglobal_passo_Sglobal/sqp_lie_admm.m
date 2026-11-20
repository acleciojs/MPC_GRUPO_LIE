function out = sqp_lie_admm(f_sym,H_sym,g_sym,Jg_sym,h_sym,Jh_sym, ...
                              delta_sym,x_sym,x0,indices_blocos,exp_cfg,opts)
% SQP_LIE_ADMM  SQP geometrica com QPs distribuidas por consenso ADMM.
%
% Assinatura:
% out = sqp_lie_admm(f_sym,H_sym,g_sym,Jg_sym,h_sym,Jh_sym, ...
%                    delta_sym,x_sym,x0,indices_blocos,exp_cfg,opts)
%
% EXPRESSOES CASADI: f_sym,g_sym,h_sym representam os PULLBACKS
% f(R_x(delta)), g(R_x(delta)), h(R_x(delta)), respectivamente.
% x_sym representa o ponto base (em coordenadas ambientes) e delta_sym
% o incremento tangente. H_sym deve ser a Hessiana de f com respeito
% a delta; Jg_sym e Jh_sym sao as Jacobianas com respeito a delta.
% Todas as expressoes podem depender de x_sym e delta_sym.
%
% min f(x)  sujeito a g(x)=0, h(x)<=0.
%
% SQP: min_d 1/2*d'*H*d + grad(f)'*d
%      s.a. g0+Jg*d=0, h0+Jh*d<=0.
%
% Os indices_blocos{i} sao os INDICES GLOBAIS do incremento delta que
% pertencem ao bloco i. Variaveis comuns a blocos consecutivos sao
% duplicadas e acopladas por CONSENSO. Toda entrada nao nula da Hessiana
% e toda linha de restricao devem caber em algum bloco, ou sera gerado
% ERRO (nenhum acoplamento sera descartado silenciosamente).
%
% A Hessiana de cada QP local recebe SOMENTE regularizacao diagonal para
% garantir PD. Assim, a QP efetiva do ADMM e a QP central de referencia
% sao as mesmas, mas podem diferir da QP global sem regularizacao.
%
% Passo SQP: x_{k+1}=passo_exponencial_sqp(x_k,d,exp_cfg); alpha=1.
% ADMM externo: resolver_admm_qps.m.
% Nao ha line search, adaptacao de rho, recuperacao ou aceitacao inexata.
% ADMM warm start por interface: transporte MPC e recenter entre SQPs.
%
% Dependencias: CasADi MATLAB e qpOASES (ou ProxQP via CasADi).

if nargin < 12 || isempty(opts), opts=struct(); end
assert(isstruct(opts) && isstruct(exp_cfg),'opts e exp_cfg devem ser structs');
assert(iscell(indices_blocos) && ~isempty(indices_blocos), ...
       'indices_blocos deve ser uma cell nao vazia');
assert(isa(delta_sym,'casadi.SX') || isa(delta_sym,'casadi.MX'), ...
       'delta_sym deve ser SX ou MX do CasADi');
assert(isa(x_sym,'casadi.SX') || isa(x_sym,'casadi.MX'), ...
       'x_sym deve ser SX ou MX do CasADi');

n=numel(delta_sym); np=numel(x_sym);
x0=full(x0(:));
assert(numel(x0)==np && all(isfinite(x0)), 'x0 deve ter numel(x_sym) valores finitos');
assert(numel(f_sym)==1, 'f_sym deve ser escalar');

if isempty(g_sym)
    if isa(delta_sym,'casadi.SX'), g_sym=casadi.SX.zeros(0,1);
    else, g_sym=casadi.MX.zeros(0,1); end
end
if isempty(h_sym)
    if isa(delta_sym,'casadi.SX'), h_sym=casadi.SX.zeros(0,1);
    else, h_sym=casadi.MX.zeros(0,1); end
end
ng=numel(g_sym); nh=numel(h_sym);
if ng==0
    if isa(delta_sym,'casadi.SX'), Jg_sym=casadi.SX.zeros(0,n);
    else, Jg_sym=casadi.MX.zeros(0,n); end
end
if nh==0
    if isa(delta_sym,'casadi.SX'), Jh_sym=casadi.SX.zeros(0,n);
    else, Jh_sym=casadi.MX.zeros(0,n); end
end
assert(isequal(size(H_sym),[n n]),'H_sym deve ser n x n');
assert(isequal(size(Jg_sym),[ng n]),'Jg_sym deve ser ng x n');
assert(isequal(size(Jh_sym),[nh n]),'Jh_sym deve ser nh x n');

% O gradiente já é gerado no builder do NMPC. Quando fornecemos
% compiled_fun, não repetimos a diferenciação simbólica por amostra.
if isfield(opts,'compiled_fun') && ~isempty(opts.compiled_fun)
    fn=opts.compiled_fun;
else
    grad_sym=jacobian(f_sym,delta_sym).';
    fn=casadi.Function('avaliar_sqp_lie_admm', {x_sym,delta_sym}, ...
        {f_sym,grad_sym,H_sym,g_sym,Jg_sym,h_sym,Jh_sym});
end

max_sqp=opt(opts,'max_sqp',1);
tol_sqp=opt(opts,'tol_sqp',1e-6);
verifica_jac=logical(opt(opts,'verificar_jacobianas',false));
tol_jac=opt(opts,'tol_jac',1e-7);
assert(max_sqp>=1 && max_sqp==floor(max_sqp),'max_sqp invalido');

admmopt=struct();
admmopt.rho=opt(opts,'rho',20);
admmopt.precond_mode=opt(opts,'precond_mode','scalar');
admmopt.precond_power=opt(opts,'precond_power',0.5);
admmopt.precond_clamp=opt(opts,'precond_clamp',[0.2 5]);
admmopt.precond_weights=opt(opts,'precond_weights',[]);
admmopt.max_iter=opt(opts,'max_admm',500);
admmopt.tol_abs=opt(opts,'tol_abs',1e-5);
admmopt.tol_rel=opt(opts,'tol_rel',1e-5);
admmopt.tol_gap_inf=opt(opts,'tol_gap_inf',1e-5);
tol_eq_lin=opt(opts,'tol_eq_lin',1e-5);
tol_ineq_lin=opt(opts,'tol_ineq_lin',1e-7);
admmopt.qp_constraint_tolerance=opt(opts,'qp_constraint_tolerance',1e-5);
admmopt.qp_optimality_tolerance=opt(opts,'qp_optimality_tolerance',1e-8);
admmopt.qp_max_iterations=opt(opts,'qp_max_iterations',800);
admmopt.qp_local_eq_tol=opt(opts,'qp_local_eq_tol',1e-6);
admmopt.qp_local_ineq_tol=opt(opts,'qp_local_ineq_tol',1e-5);
admmopt.qp_restart_on_failure=opt(opts,'qp_restart_on_failure',true);
admmopt.solver=opt(opts,'qp_solver','qpoases');
admmopt.admm_warm_local_qp=logical(opt(opts,'admm_warm_local_qp',true));
admmopt.admm_seed=[];
admm_warm=logical(opt(opts,'admm_warm_start',true));
seed_primeira=opt(opts,'admm_seed',[]);
admmopt.verbose=logical(opt(opts,'verbose_admm',false));
admmopt.verbose_every=opt(opts,'verbose_every',50);
admmopt.reg_base=opt(opts,'reg_base',1e-4);
admmopt.reg_floor=opt(opts,'reg_floor',1e-5);
assert(admmopt.rho>0 && admmopt.reg_floor>0 && admmopt.reg_base>=0, ...
       'rho e regularizacao devem ser validos');

lb=opt(opts,'lb_delta',-inf(n,1));
ub=opt(opts,'ub_delta', inf(n,1));
lb=lb(:);ub=ub(:);
assert(numel(lb)==n && numel(ub)==n && all(lb<=ub), ...
       'Limites delta invalidos');

out=struct();out.status='max_sqp';out.message='';out.x=x0;
out.delta=zeros(n,1);
% Inicializar os campos do historico antes da primeira atribuicao.
% struct([]) sem campos pode gerar 'Subscripted assignment between
% dissimilar structures' em algumas versoes de MATLAB.
out.sqp=struct('iter',{},'f',{},'delta',{},'admm',{}, ...
    'reg_max',{},'resid_eq',{},'viol_ineq',{});
out.completed_sqp=0;
out.f_initial=NaN;out.f_final=NaN;out.last_blocks=[];
out.admm_final=[];out.admm_base=[];out.admm_seed_first=[];
prev_admm=[];prev_base=[];
out.central_comparison=struct('delta_central',{}, ...
    'delta_admm',{},'erro_relativo',{},'defeito_admm',{}, ...
    'defeito_central',{});
x=x0;
for k=1:max_sqp
    [f0v,gv,Hv,g0v,Jgv,h0v,Jhv]=fn(x,zeros(n,1));
    f0=full(f0v); grad=full(gv); H=full(Hv);
    g0=full(g0v); Jg=full(Jgv); h0=full(h0v); Jh=full(Jhv);
    grad=grad(:);g0=g0(:);h0=h0(:);
    assert(all(isfinite([f0;grad;H(:);g0;Jg(:);h0;Jh(:)])), ...
           'CasADi gerou NaN/Inf ao avaliar o problema');
    H=0.5*(H+H.'); % Somente a parte simetrica define a forma quadratica
    if k==1, out.f_initial=f0; end

    if verifica_jac
        [gcalc,hcalc]=jac_check(f_sym,g_sym,h_sym,delta_sym,x_sym,x);
        assert(norm(Jg-gcalc,inf)<tol_jac*max(1,norm(gcalc,inf)), ...
            'Jg_sym nao coincide com a derivada de g em delta=0');
        assert(norm(Jh-hcalc,inf)<tol_jac*max(1,norm(hcalc,inf)), ...
            'Jh_sym nao coincide com a derivada de h em delta=0');
    end

    % As linhas das restricoes sao atribuidas integralmente aos blocos.
    % As copias de interface sao ligadas por consenso (ADMM externo).
    blocos=decompor_qp_em_blocos(H,grad,g0,Jg,h0,Jh, ...
                                indices_blocos,lb,ub,admmopt);
    admmopt.admm_seed=[];
    if admm_warm
        if k==1
            admmopt.admm_seed=seed_primeira;
            out.admm_seed_first=seed_primeira;
        elseif ~isempty(prev_admm) && prev_admm.converged
            % NOVA base SQP ja inclui o passo exponencial anterior.
            % Transportar duais; primal = log(new_base^{-1} * new_base)=0.
            na=exp_cfg.n_state;nu=exp_cfg.n_control;
            N=exp_cfg.num_nodes-1;
            Xold=reshape(prev_base(1:na*(N+1)),na,N+1);
            Xnew=reshape(x(1:na*(N+1)),na,N+1);
            Unew=reshape(x(na*(N+1)+1:end),nu,N);
            try
                admmopt.admm_seed=preparar_warm_admm(prev_admm, ...
                    Xold,Xnew,Xnew,Unew,Unew,indices_blocos,N/numel(blocos),0,opts);
            catch ME
                warning('Warm SQP descartado: %s',ME.message);
                admmopt.admm_seed=[];
            end
        end
    end
    x_base_qp=x;
    [d,info]=resolver_admm_qps(blocos,admmopt);

    log=struct('iter',k,'f',f0,'delta',d,'admm',info, ...
        'reg_max',max(cellfun(@(b)b.reg_delta,blocos)), ...
        'resid_eq',NaN,'viol_ineq',NaN);
    if ~info.converged
        out.status='admm_nao_convergiu';
        out.message=sprintf('SQP %d: ADMM parou em %d iteracoes (%s)', ...
                            k,info.iter,info.reason);
        out.sqp(k)=log;out.last_blocks=blocos;
        warning('%s',out.message);
        break;
    end

    log.resid_eq=norm(Jg*d+g0,inf);
    log.viol_ineq=max([0;Jh*d+h0]);
    % Uma QP local pode ser viavel e o vetor global reconstruido nao ser:
    % as copias das interfaces diferem enquanto o consenso e inexato.
    % Nao aplicar passo exponencial com dinâmica linearizada inadequada.
    if log.resid_eq>tol_eq_lin || log.viol_ineq>tol_ineq_lin
        out.status='qp_global_inviavel';
        out.message=sprintf(['SQP %d: delta reconstruido viola restricoes: ' ...
            '||Jg*d+g||inf=%.3e (tol %.3e), ineq=%.3e (tol %.3e)'], ...
            k,log.resid_eq,tol_eq_lin,log.viol_ineq,tol_ineq_lin);
        out.sqp(k)=log;out.last_blocks=blocos;
        warning('%s',out.message);
        break;
    end
    if logical(opt(opts,'compare_central',false))
        QPc=montar_qp_central_de_blocos(blocos,n);
        try
            cmp=qp_casadi_criar(QPc.H,QPc.Aeq,QPc.beq, ...
                QPc.Aineq,QPc.bineq,QPc.lb,QPc.ub, ...
                admmopt.solver,admmopt.qp_max_iterations,'qp_central_cmp');
            [dc,~,~,~]=qp_casadi_resolver(cmp,QPc.f);
            out.central_comparison(k)=struct('delta_central',dc, ...
                'delta_admm',d,'erro_relativo',norm(dc-d)/max(1,norm(dc)), ...
                'defeito_admm',norm(QPc.Aeq*d-QPc.beq,inf), ...
                'defeito_central',norm(QPc.Aeq*dc-QPc.beq,inf));
        catch ME
            warning('Comparacao central indisponivel via CasADi: %s',ME.message);
        end
    end

    % Cache dual corresponde à BASE em que a QP foi linearizada.
    % Guardar antes de atualizar a trajetoria via passo exponencial.
    out.admm_final=info;
    out.admm_base=x_base_qp;
    prev_admm=info;
    prev_base=x_base_qp;

    % CHAMADA EXTERNA: multiplicacao exponencial, com alpha_SQP=1.
    x=passo_exponencial_sqp(x,d,exp_cfg);
    out.x=x;out.delta=d;out.completed_sqp=k;
    out.last_blocks=blocos;out.sqp(k)=log;
    if logical(opt(opts,'verbose',true))
        fprintf(['SQP=%d | ||d||inf=%.3e | ADMM=%d | r/eps=%.3g ' ...
                 's/eps=%.3g | eq_lin=%.3e | h_lin=%.3e\n'], ...
                 k,norm(d,inf),info.iter,info.r_pr/max(eps,info.eps_pr), ...
                 info.r_du/max(eps,info.eps_du),log.resid_eq,log.viol_ineq);
    end
    if norm(d,inf)<tol_sqp
        out.status='convergiu';out.message='Incremento SQP pequeno';
        break;
    end
end
fv=fn(x,zeros(n,1));
out.f_final=full(fv);
end

function [Jg,Jh]=jac_check(f,g,h,delta,xsym,x)
% O argumento f e mantido na assinatura por clareza, sem uso nesta checagem.
unused=f; %#ok<NASGU>
fun=casadi.Function('verificar_jac_lie', {xsym,delta}, ...
    {jacobian(g,delta),jacobian(h,delta)});
[gg,hh]=fun(x,zeros(numel(delta),1));
Jg=full(gg);Jh=full(hh);
end

function val=opt(s,k,def)
if isfield(s,k) && ~isempty(s.(k)),val=s.(k);else,val=def;end
end
