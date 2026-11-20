% TESTAR_FUNCOES_SYMBOLIC_USUARIO
% Teste independente de modelo para as duas rotinas CasADi.
% Em SO(2): f(R)=1-R11 => d2 f(R exp(eta J))/deta2|0=1
% e g(R)=R21 => d g(R exp(eta J))/deta|0=1.
import casadi.*
R=SX.sym('R_teste',2,2);
w=SX.sym('w_teste',1);
S=[0,-w;w,0];
f=1-R(1,1);
g=R(2,1);
H=symbolic_Hessian_simetrica(f,{R},S,w,0);
J=symbolic_Jacobian(g,{R},S,w);
fun=Function('testar_derivadas_so2_usuario',{R},{H,J});
[Hv,Jv]=fun(eye(2));
Hv=full(Hv);Jv=full(Jv);
fprintf('SO2: Hessiana=%.12g (esperado 1); Jacobiana=%.12g (esperado 1)\n',Hv,Jv);
assert(abs(Hv-1)<1e-10 && abs(Jv-1)<1e-10, ...
       'Derivadas geometricas do teste SO2 incorretas');
