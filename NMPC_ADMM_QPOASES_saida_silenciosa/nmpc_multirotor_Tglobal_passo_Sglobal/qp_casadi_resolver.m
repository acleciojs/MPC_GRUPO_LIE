function [x,lam_a,lam_x,stats] = qp_casadi_resolver(cache,g,x0,lam_a0,lam_x0)
% QP_CASADI_RESOLVER  Resolve QP usando cache de CasADi conic.
% A instancia e reutilizada para aproveitar a estrutura e o hotstart.
n=cache.n;m=cache.m;
g=full(g(:));
assert(numel(g)==n && all(isfinite(g)),'Gradiente linear invalido');
if nargin<3 || isempty(x0),x0=zeros(n,1);end
if nargin<4 || isempty(lam_a0),lam_a0=zeros(m,1);end
if nargin<5 || isempty(lam_x0),lam_x0=zeros(n,1);end
x0=full(x0(:));lam_a0=full(lam_a0(:));lam_x0=full(lam_x0(:));
assert(numel(x0)==n && numel(lam_a0)==m && numel(lam_x0)==n, ...
    'Dimensoes de warm start incorretas');
assert(all(isfinite([x0;lam_a0;lam_x0])), ...
    'Warm start com valores nao finitos');
sol=cache.solver('h',cache.H,'g',g,'a',cache.A, ...
    'lba',cache.lba,'uba',cache.uba, ...
    'lbx',cache.lbx,'ubx',cache.ubx, ...
    'x0',x0,'lam_a0',lam_a0,'lam_x0',lam_x0);
stats=cache.solver.stats();
if isfield(stats,'success') && ~stats.success
    msg='Retorno sem convergencia';
    if isfield(stats,'return_status')
        msg=char(stats.return_status);
    end
    error('%s: %s',cache.backend,msg);
end
x=full(sol.x);lam_a=full(sol.lam_a);lam_x=full(sol.lam_x);
assert(numel(x)==n && numel(lam_a)==m && numel(lam_x)==n && ...
    all(isfinite([x(:);lam_a(:);lam_x(:)])), ...
    'QP retornou valores invalidos');
x=x(:);lam_a=lam_a(:);lam_x=lam_x(:);
end
