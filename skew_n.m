function S = skew_n(omega)
    % Generic skew-symmetric matrix builder for so(n)
    % omega: vector of length n*(n-1)/2
    % Returns: n x n skew-symmetric matrix
    
    import casadi.*
    
   S = [0,-omega(3),omega(2);omega(3),0,-omega(1);-omega(2),omega(1),0];


end