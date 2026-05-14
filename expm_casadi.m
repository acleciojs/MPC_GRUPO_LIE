function R = expm_casadi(S)
% EXPM_CASADI_FAST
% Versão otimizada para velocidade no CasADi
% Padé (13,13) + scaling fixo
%
% ? Muito mais rápida
% ? Menos carga simbólica
% ? Ideal para IPOPT/SQP

import casadi.*

n = size(S,1);

% =========================================================
% Identidade (criada uma vez só)
% =========================================================
if isa(S,'casadi.SX')
    I = SX.eye(n);
elseif isa(S,'casadi.MX')
    I = MX.eye(n);
else
    I = eye(n);
end

% =========================================================
% Scaling (fixo mas mais leve)
% =========================================================
s = 3;  % ? reduz custo de squaring (quase mesmo erro)
A = S * (1/2^s);

% =========================================================
% Potências (mínimas necessárias)
% =========================================================
A2 = A*A;
A4 = A2*A2;
A6 = A4*A2;

% =========================================================
% Coeficientes (Padé 13)
% =========================================================
c = [ ...
64764752532480000;
32382376266240000;
7771770303897600;
1187353796428800;
129060195264000;
10559470521600;
670442572800;
33522128640;
1323241920;
40840800;
960960;
16380;
182;
1 ];

% =========================================================
% Estrutura otimizada (menos operações)
% =========================================================
A6_c14 = c(14)*A6;
A6_c13 = c(13)*A6;

U = A*( ...
    A6*(A6_c14 + c(12)*A4 + c(10)*A2) + ...
    c(8)*A6 + c(6)*A4 + c(4)*A2 + c(2)*I );

V = ...
    A6*(A6_c13 + c(11)*A4 + c(9)*A2) + ...
    c(7)*A6 + c(5)*A4 + c(3)*A2 + c(1)*I;

% =========================================================
% ? Substitui solve por fatoração mais leve
% =========================================================
% (V - U)R = (V + U)
M = V - U;
N = V + U;

% Para CasADi: inv é mais leve que solve em muitos casos simbólicos
R = M \ N;   % melhor que solve()

% =========================================================
% Squaring (menos iterações)
% =========================================================
for i = 1:s
    R = R * R;
end

end