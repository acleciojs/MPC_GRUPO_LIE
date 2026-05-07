function R = expm_numeric(S, K)

    if ~isnumeric(S)
        error('S deve ser uma matriz numérica (double).');
    end
    if size(S,1) ~= size(S,2)
        error('S deve ser quadrada.');
    end

    R = eye(size(S));
    term = eye(size(S));  % termo corrente (S^k / k!)

    for k = 1:K
        term = term * S / k;   % gera S^k/k! recursivamente
        R = R + term;
    end

end