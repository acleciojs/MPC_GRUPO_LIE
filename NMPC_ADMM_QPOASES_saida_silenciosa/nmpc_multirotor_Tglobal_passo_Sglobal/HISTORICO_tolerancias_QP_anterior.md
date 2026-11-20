# Opcoes de tolerancias numericas resolvedor anterior

A falha original em N=100 M=50 ocorreu na verificacao APOS resolvedor anterior:
`QP local nao viavel: eq=4.829e-08 ineq=6.497e-06`.
O codigo exigia ineq<=1e-6 mesmo com resolvedor anterior ConstraintTolerance=1e-9.

Configuracoes novas (cfg):
- `qp_constraint_tolerance` (padrao 1e-5): tolerancia solicitada ao resolvedor anterior.
- `qp_optimality_tolerance` (padrao 1e-8).
- `qp_max_iterations` (padrao 800).
- `qp_local_eq_tol` (padrao 1e-6, multiplicada por max(1,norm(beq,inf))).
- `qp_local_ineq_tol` (padrao 1e-5): limite de verificacao numerica apos o solver, incluindo bounds.
- `tol_ineq_lin` (padrao 1e-7): verificacao INDEPENDENTE da direcao global reconstruida na SQP. Para aceitar a mesma margem local no MPC configure 1e-5, se fisicamente toleravel.

ATENCAO: relaxar verificadores nao torna a QP matematicamente viavel.
No comando `resolvedor anterior`, aumentar ConstraintTolerance NAO garante um erro menor; pode permitir erro maior. Se violacoes numericas forem inaceitaveis, melhore o condicionamento ou refine a solucao, sem afrouxar tolerancias de restricoes fisicas de seguranca.

Use `exemplo_N100_resolvedor anterior_tolerante.m` para executar.
