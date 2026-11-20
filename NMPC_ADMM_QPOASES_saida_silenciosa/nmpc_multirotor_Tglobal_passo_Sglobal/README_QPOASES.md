# NMPC com SQP geometrica + ADMM precondicionado + qpOASES

Este pacote **nao usa quadprog**. O backend principal dos subproblemas locais e
`casadi.conic(...,'qpoases',...)`, e o backend alternativo configuravel e
`proxqp` **somente se o seu CasADi incluir o plugin**.

## Passo a passo

1. Adicione esta pasta ao path MATLAB e mantenha CasADi e as funcoes do
   modelo `Init`, `Equilibrio`, `IntegrationEuler`, `Attitude_error` no path.
2. Verifique o plugin: `import casadi.*; disp(has_conic('qpoases'));`
3. Rode `testar_admm_precondicionado_qp` e `testar_warm_admm`.
4. Rode `exemplo_N200_M50_qpoases`.

## O que mudou

- Removida toda chamada ao quadprog (inclusive na comparacao central opcional).
- Hessiana/estrutura de restricoes sao construidas somente uma vez por bloco
  para uma chamada ADMM, usando o nivel baixo `conic` do CasADi.
- Uma instancia de qpOASES permanece viva durante as iteracoes do ADMM.
- Usa solucao primal e multiplicadores das restricoes da QP anterior
  (`x0`, `lam_a0`, `lam_x0`) quando validos.
- Em caso de falha numerica: reinicia apenas o bloco com maior limite de
  iteracoes do qpOASES (uma tentativa), com erro explicito se falhar novamente.
- Mantidos ADMM precondicionado P_i, warm start entre SQPs/amostras, passo
  exponencial e IntegrationEuler da planta.

`cfg.qp_max_iterations` passa a configurar o limite de trocas de conjunto
ativo `nWSR` do qpOASES, nao MaxIterations do MATLAB. A tolerancia
`cfg.qp_local_ineq_tol` controla apenas a validacao local, nao a precisao
matematica pedida ao solver. Para diagnostico `out.sqp{t}.sqp(end).admm`
registra `qp_calls`, `qp_restarts` e `qp_iter_total`.

**Limitacoes:** este ambiente nao tem MATLAB nem o modelo externo do VANT.
Os testes do CasADi qpOASES foram feitos em um problema QP numerico
independente, e o NMPC N=200/M=50 nao foi executado aqui. Reiniciar o
backend nao corrige inviabilidade real; a qualidade do SQP/ADMM ainda deve
ser validada na planta.


## Evitar o banner qpOASES em cada amostra MPC

`qp_casadi_criar.m` ja define `conf.printLevel='none'` e
`conf.print_time=false`. Agora o construtor CasADi e executado dentro
`evalc` para capturar a saida de inicializacao visivel ao MATLAB, sem
alterar a formulacao ou as mensagens de erro retornadas via excecao.

**Limitacao**: algumas distribuicoes binaras de CasADi imprimem esse
aviso por stdout C/C++, que pode contornar `evalc`. Nesse caso, a
solucao definitiva e usar CasADi compilado com
`-DWITH_NO_QPOASES_BANNER=ON` (disponivel nas versoes que oferecem
essa opcao CMake). O equivalente qpOASES puro e compilar com
`-D__NO_COPYRIGHT__`. Nao confunda esse aviso com os logs de iteracao:
`printLevel='none'` ja desliga os logs normais do solver.

Evite `diary off` ou `warning off all`: nao impedem necessariamente
stdout nativo e escondem mensagens uteis do MPC.
