import casadi.*

%% variável simbólica
R = SX.sym('R',3,3);
R_list = {R};

%% omega simbólico
omega = SX.sym('omega',3);

%% definir S(omega)
S_sym = [     0      -omega(3)  omega(2);
           omega(3)     0      -omega(1);
          -omega(2)  omega(1)     0 ];

%% função f
f = trace(R);

%% chamar
J =Jacobian_f(f, R_list, S_sym, omega)

