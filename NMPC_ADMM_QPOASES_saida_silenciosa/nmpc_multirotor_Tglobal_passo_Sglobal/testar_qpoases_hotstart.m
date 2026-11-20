function resultado = testar_qpoases_hotstart()
% Teste autocontido de qpOASES via CasADi conic, sem modelo do VANT.
% Mesmo tamanho de subproblema de N=200, M=50: 132 variaveis e 80 igualdades.
import casadi.*
n=132;me=80;mi=26;
H=spdiags(linspace(1,3,n)',0,n,n);
Aeq=[speye(me),sparse(me,n-me)];beq=zeros(me,1);
Aineq=[sparse(mi,me),speye(mi),sparse(mi,n-me-mi)];
bineq=ones(mi,1);lb=-5*ones(n,1);ub=5*ones(n,1);
cache=qp_casadi_criar(H,Aeq,beq,Aineq,bineq,lb,ub, ...
                     'qpoases',2000,'teste_hotstart_qpoases');
x=zeros(n,1);la=zeros(me+mi,1);lx=zeros(n,1);
gap=zeros(5,1);wsr=zeros(5,1);
for k=1:5
    g=-0.04*ones(n,1)+(k-1)*0.001*(1:n)';
    [x,la,lx,st]=qp_casadi_resolver(cache,g,x,la,lx);
    gap(k)=max([norm(Aeq*x-beq,inf);0;Aineq*x-bineq;lb-x;x-ub]);
    if isfield(st,'iter_count'), wsr(k)=double(st.iter_count);end
    assert(gap(k)<1e-6,'QP via qpOASES violou restricoes');
    fprintf('QP %d | nWSR=%d | violacao=%.2e\n',k,wsr(k),gap(k));
end
resultado=struct('iter_qp',wsr,'violacao',gap,'ultimo_x',x);
end
