**VERSAO ATUAL:** consulte primeiro `README_QPOASES.md`; todas as QPs usam CasADi conic.

# NMPC multirrotor - malha fechada com IntegrationEuler

Esta versao mantem as derivadas geometricas por T_GLOBAL/S_GLOBAL e o resolvedor SQP+ADMM modular. O controlador passa a usar **explicitamente** o horizonte anterior para warm start do **MPC somente**. Cada QP do ADMM e resolvida com z, w e duais inicializados em zero em `resolver_admm_qps.m`.

## Executar

Coloque esta pasta no path, alem das dependencias originais: `Init.m`, `Equilibrio.m`, `IntegrationEuler.m`, `Attitude_error.m`, CasADi e `qpOASES` pelo CasADi.

```matlab
clear functions; clc; close all;
addpath('nmpc_multirotor_Tglobal_malha_fechada');
cfg = struct();
cfg.N = 40;
cfg.M = 10;
cfg.dt = 0.01;
cfg.T_sim = 20;                  % aumente para 100 (1 s), 500, ...
cfg.max_sqp = 1;
cfg.max_admm = 2500;
cfg.rho = 20;
cfg.tol_abs = 1e-5;
cfg.tol_rel = 1e-5;
cfg.tol_gap_inf = 1e-7;
cfg.tol_eq_lin = 1e-5;
cfg.tol_ineq_lin = 1e-7;
cfg.qp_solver = 'qpoases';
cfg.warm_start = true;
cfg.warm_state_mode = 'shift';    % 'shift' (estados+controles) ou 'rollout'
cfg.plot_results = true;
out = nmpc_multirotor_usando_sqp_lie_admm(cfg);
```

## Sequencia de cada instante

- `t=1`: nominal inicial `U=Equilibrio(x_ref)` e estados obtidos com `IntegrationEuler`.
- `t>1`: deslocar **apenas** as trajetorias primais otimizadas no instante anterior: `Uwarm=[Uopt(:,2:end), Uopt(:,end)]`; `Xwarm(:,1)=x_real`; `Xwarm(:,2:N)=Xopt(:,3:N+1)`; ultimo no obtido com `IntegrationEuler(Xwarm(:,N),Uwarm(:,N))`. O modelo pode ter defeito residual se o horizonte anterior nao era exatamente viavel. Alternativa `cfg.warm_state_mode='rollout'` re-integra todos os estados com os controles deslocados.
- Calcular `base=[Xwarm(:); Uwarm(:)]` e chamar `sqp_lie_admm`. **Nao passa z/gamma/delta do ADMM anterior**; o ADMM e reinicializado de zero em cada QP.
- Se a SQP conclui o passo, extrair `u_aplicar=Uopt(:,1)` e integrar **a planta real**: `x_real_next=IntegrationEuler(x_real,u_aplicar,dt)` atraves da funcao CasADi compilada que possui essa expressao. A planta nunca recebe diretamente `Xopt(:,2)`.
- `out.x` armazena estados reais da planta, `out.u` comandos aplicados, `out.Xpred` e `out.Upred` contem todas as predicoes otimizadas, e `out.Xwarm` e `out.Uwarm` os chutes nominais. `out.def_pred` e `out.def_warm` sao defeitos dinamicos nao lineares da predicao e do chute nominal; o controle aplicado nao muda essa medicao.

## Observacoes numericas

- ADMM convergente na QP linearizada **nao** garante dinamica nao linear exata no horizonte depois do passo exponencial. Monitorar `out.def_pred` e o erro real `out.err` separadamente.
- Um passo de MPC com `dt=0.01` so avanca 0.01 segundo; `out.err` pode quase nao mudar na primeira amostra.
- O codigo nao projeta estados fora do grupo; se `IntegrationEuler` utilizar integracao nao geometrica para rotacoes, o controlador interrompe com erro explicito. Atualize a integracao de rotacoes para manter SO(3) e SO(2)^4.
- Nao ha warm-start dos duais ou consenso ADMM, e nem chamada a `solver_IPM`.
- O tempo de execucao pode ser alto: seu diagnostico anterior precisou de 1840 iteracoes ADMM na primeira QP.


## Atualizacao exponencial via o mesmo S_GLOBAL das derivadas

A nova versao compila duas funcoes CasADi **uma unica vez** no gerador:

```matlab
exp_cfg.T_global_fun = Function('T_global_base_retracao',{X},{Tbase});
exp_cfg.S_global_fun = Function('S_global_passo_retracao',{omega_global},{S_global});
```

A funcao `passo_exponencial_sqp(x,delta,exp_cfg)` agora avalia de fato
`S_global_fun(delta)` e `T_global_fun(x)`; para cada no executa
`Tnext = Tk*expm(Sk)` e desempacota os 30 estados/8 controles.
A abordagem por blocos implementa **exatamente**
`Tglobal_next = Tglobal * expm(S_global(delta))` porque o grupo produto
possui estrutura bloco-diagonal. **Nao** sao usadas bases de Lie
redefinidas/independentes no passo do multirrotor; nao ha expansao de Taylor
no passo numerico. `expm_casadi` de ordem 2 continua reservado apenas
as derivadas no incremento nulo.

O exemplo independente `exemplo_sqp_lie_admm_SO2` continua compativel
com o modo antigo por bases quando `exp_cfg.mode` nao esta presente.

Para testar a equivalencia do passo numerico sem o modelo fisico, execute
`testar_passo_exponencial_Sglobal` no MATLAB com CasADi no path.

## NOVO — ADMM precondicionado nas interfaces (opcional)

Nesta revisao foi alterado **apenas** o resolvedor ADMM e a passagem
`cfg -> opts -> admmopt`. A formacao das derivadas `T_GLOBAL/S_GLOBAL`, a
SQP geometrica, o passo `S_GLOBAL(delta)` e a planta `IntegrationEuler`
foram preservados. Nao existe warm start do ADMM nesta versao, apenas
warm start das trajetorias do MPC, como no ZIP original.

Para usar:

```matlab
cfg=struct();
cfg.N=40; cfg.M=10; cfg.T_sim=1; cfg.dt=0.01;
cfg.max_sqp=1; cfg.max_admm=2500; cfg.rho=20;
cfg.tol_abs=1e-5; cfg.tol_rel=1e-5;
cfg.tol_gap_inf=1e-7; cfg.tol_eq_lin=1e-5;
cfg.precond_mode='curvature';   % 'scalar' = metodo anterior
cfg.precond_power=0.5;
cfg.precond_clamp=[0.2 5];
cfg.qp_solver='qpoases';
out=nmpc_multirotor_usando_sqp_lie_admm(cfg);
```

Outros modos: `cfg.precond_mode='manual'` com
`cfg.precond_weights` vetor positivo de tamanho 20 (repete por interface)
ou `cell(M-1,1)` com um vetor positivo em cada interface.

A interface i usa `P_i = rho*diag(p_i)`, SPD. No modo `curvature`, para cada
coordenada compartilhada, calculam-se `a=diag(Le*H_i*Le')` e
`b=diag(Ls*H_(i+1)*Ls')`, `raw=sqrt(a.*b)`, e
`p_i = clip((raw/median(raw))^power,[0.2,5])`.
Isto utiliza uma **aproximacao diagonal da curvatura**, nao a Hessiana
reduzida exata nem a sensibilidade da interface; seu ganho depende do caso.

As quatro alteracoes do ADMM sao:

1. Hessiana local `Haug=H+L'*P*L` e custo linear `caug=c+L'*(gamma-P*z)`.
2. Consenso `z=(xe+xs)/2+(gammae+gammas)./(2*diag(P))`.
3. Multiplicadores `gamma += P*(x-z)`.
4. Residuo dual calculado com `Le'*(P*Delta_z)` e `Ls'*(P*Delta_z)`.

`Haug` e os parametros constantes do qpOASES sao montados uma vez por QP.
As tolerancias originais e o criterio absoluto `tol_gap_inf` sao mantidos.
A QP centralizada e a regularizacao local originais nao sao modificadas;
somente a coordenacao recebe a metrica diagonal.

Testes:

```matlab
testar_admm_precondicionado_qp;  % QP pequena, sem modelo do VANT
exemplo_ADMM_precondicionado;    % MPC N=40, M=10
comparar_ADMM_precondicionado;   % compara os dois modos
```

Nao ha promessa de reducao de iteracoes: por exemplo, um peso diagonal
calculado da Hessiana nao modela as restricoes dinamicas ou o conjunto ativo.
No MATLAB, compare numero de iteracoes, tempo do ADMM e gap para as mesmas
tolerancias. Somente aceite o passo SQP quando a QP reconstruida for viavel.

## NOVO — Warm start completo do ADMM precondicionado (novembro/2026)

O projeto inclui `preparar_warm_admm.m`, utilizado em duas camadas:

- **Entre amostras do NMPC**: desloca os multiplicadores não escalados das
  interfaces um instante à frente, por interpolação temporal linear entre
  interfaces; transporta as três componentes de SO(3) para a nova base da
  perturbação à direita e aplica amortecimento conservador. Reconstrói o
  vetor primal no novo ponto-base, por diferença euclidiana, logaritmo de
  SO(3) e diferença angular de SO(2). O alvo primal vem da predição ótima
  anterior deslocada. Se a nova nominal é essa própria trajetória, o
  incremento inicial é zero (não se aplica duas vezes a correção anterior).
- **Entre iterações SQP**: transporta os multiplicadores para a nova base
  após o passo exponencial; recentra o primal na nova nominal, sem reutilizar
  diretamente a direção antiga.
- **Dentro das iterações ADMM**: reaproveita `w{i}` como chute do resolvedor
  local se ele satisfizer restrições locais. Isso é passado ao qpOASES e à
  variante opcional `qpoases`. O `qpoases` com
  `interior-point-convex` *não aproveita* esse x0, porém o warm de `z` e
  dos multiplicadores continua plenamente ativo.

As sementes são validadas quanto a dimensão e finitude. As duas cópias de
cada interface recebem multiplicadores opostos, mantendo
`gamma_e{i}+gamma_s{i}=0`. Todos os multiplicadores são **não escalados**:
na mudança da métrica `P_i`, não multiplicar ou dividir `gamma` por `rho`.
A aproximação do transporte temporal dos duais não é exata e pode reduzir
performance caso a trajetória mude muito; por isso o amortecimento é
configurável e sempre é possível desligar o warm.

Configuração recomendada inicial:

```matlab
cfg.warm_start = true;
cfg.warm_state_mode = 'rollout';
cfg.admm_warm_start = true;
cfg.admm_warm_dual_damping = 0.95;
cfg.admm_warm_primal_damping = 1.0;
cfg.admm_warm_primal_max_inf = 0.5;
cfg.admm_warm_local_qp = true;
cfg.precond_mode = 'curvature';
cfg.qp_solver = 'qpoases'; % seguro; opção 'qpoases' para testar hot x0
```

Arquivos adicionais:
- `preparar_warm_admm.m` — transporte geométrico e temporal do primal/dual;
- `testar_warm_admm.m` — testes em QPs convexas simples e transporte SO(3);
- `exemplo_ADMM_precondicionado_warm.m` — script de malha fechada.
- `comparar_ADMM_warm_start.m` — benchmark warm ligado/desligado.

Teste primeiro `testar_warm_admm` (requer `qpoases`), depois
`exemplo_ADMM_precondicionado_warm` (requer dependências da planta e CasADi).
Compare, na mesma QP e mesmas tolerâncias, `cfg.admm_warm_start=false` e
`true`, registrando `out.admm_iters`, `out.admm_warm_used` e `out.gap`.
Essa implementação ainda não foi executada dentro do MATLAB neste ambiente.
