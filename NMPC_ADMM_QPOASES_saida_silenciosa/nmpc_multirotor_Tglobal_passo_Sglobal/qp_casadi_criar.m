function cache = qp_casadi_criar(H,Aeq,beq,Aineq,bineq,lb,ub,backend,max_iter,nome)
% QP_CASADI_CRIAR  Cria UMA instancia CasADi conic para QP convexa.
%   min 0.5*x'*H*x + g'*x
%   Aeq*x=beq, Aineq*x<=bineq, lb<=x<=ub.
% H, A e os limites permanecem constantes durante o ADMM.
% backend = 'qpoases' (padrao) ou 'proxqp' (se instalado no CasADi).
import casadi.*
if nargin<8 || isempty(backend),backend='qpoases';end
if nargin<9 || isempty(max_iter),max_iter=2000;end
if nargin<10 || isempty(nome),nome='qp_casadi';end
backend=lower(char(backend));
assert(ismember(backend,{'qpoases','proxqp'}),'Backend nao suportado');
H=sparse(0.5*(H+H.'));
n=size(H,1);
assert(size(H,2)==n,'H deve ser quadrada');
if isempty(Aeq),Aeq=sparse(0,n);beq=zeros(0,1);end
if isempty(Aineq),Aineq=sparse(0,n);bineq=zeros(0,1);end
A=sparse([Aeq;Aineq]);
me=size(Aeq,1); mi=size(Aineq,1);
assert(numel(beq)==me && numel(bineq)==mi,'Dimensao de b incorreta');
assert(size(A,2)==n,'A tem colunas diferentes de H');
if isempty(lb),lb=-inf(n,1);end
if isempty(ub),ub= inf(n,1);end
lb=full(lb(:));ub=full(ub(:));
assert(numel(lb)==n && numel(ub)==n && all(lb<=ub), ...
    'Limites de variaveis invalidos');
% Matriz de restricoes empilhadas: igualdades + desigualdades.
lba=[full(beq(:));-inf(mi,1)];
uba=[full(beq(:));full(bineq(:))];
Hd=DM(H);Ad=DM(A);
qp=struct('h',Hd.sparsity(),'a',Ad.sparsity());
conf=struct('error_on_fail',false,'print_time',false);
if strcmp(backend,'qpoases')
    conf.printLevel='none';
    conf.nWSR=double(max_iter);
    conf.enableEqualities=true;
    % Evite forcar regularizacao interna: para Hessianas diagonais/identidade
    % o qpOASES pode recusar essa combinacao mesmo em QPs convexas.
    conf.enableRegularisation=false;
end
try
    % O plugin qpOASES pode imprimir o aviso de copyright ao CRIAR
    % cada instancia, mesmo com printLevel='none'. Capturamos apenas
    % a saida da construcao; erros continuam propagados pelo catch.
    % Nota: algumas builds de CasADi escrevem diretamente no stdout C;
    % nesse caso evalc nao consegue capturar o banner nativo.
    if strcmp(backend,'qpoases')
        texto_inicializacao = evalc('solver=conic(nome,backend,qp,conf);'); %#ok<NASGU>
    else
        solver=conic(nome,backend,qp,conf);
    end
catch ME
    error(['Nao foi possivel criar %s via CasADi: %s. ' ...
        'Verifique a instalacao do plugin e as matrizes da QP.'], ...
        backend,ME.message);
end
cache=struct('solver',solver,'H',Hd,'A',Ad,'lba',lba,'uba',uba, ...
    'lbx',lb,'ubx',ub,'n',n,'m',me+mi,'backend',backend);
end
