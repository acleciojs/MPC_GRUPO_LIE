function E=expm_casadi(S)
% EXPM_CASADI: jato de segunda ordem usado EXCLUSIVAMENTE nas
% rotinas symbolic_Hessian_simetrica/symbolic_Jacobian em xi=0.
%
% Exponencial exata: exp(S)=I+S+S^2/2+O(||S||^3).
% Logo, valor, Jacobiana e Hessiana em S=0 sao EXATOS.
% NAO utilizar esta aproximacao para atualizar o estado fora de xi=0!
% O passo externo da SQP usa expm numerica exata.
    import casadi.*
    n=size(S,1);
    if isa(S,'casadi.SX')
        I=SX.eye(n);
    elseif isa(S,'casadi.MX')
        I=MX.eye(n);
    else
        I=eye(n);
    end
    E=I+S+0.5*(S*S);
end
