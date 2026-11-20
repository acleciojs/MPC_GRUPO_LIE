function y = project_euler_state_symbolic(x)
% PROJECT_EULER_STATE_SYMBOLIC  Aproximacao suave da projecao polar.
% Usar identicamente no integrador simbolico e no residual da SQP.
% SO3: 4 iteracoes Newton-Schulz (acuradas perto de SO(3)); SO2: normalizacao.
% Nao utilizar se IntegrationEuler ja for integrador no grupo.
import casadi.*
y=x(:);
assert(numel(y)==30,'O integrador deve retornar 30 componentes');
Rp=reshape(y(14:22),3,3);
I3=SX.eye(3);
for j=1:4
    Rp=0.5*Rp*(3*I3-Rp'*Rp);
end
y(14:22)=reshape(Rp,9,1);
for j=0:3
    idx=23+2*j;
    a=y(idx); b=y(idx+1);
    den=sqrt(a*a+b*b+1e-24);
    y(idx)=a/den;
    y(idx+1)=b/den;
end
end
