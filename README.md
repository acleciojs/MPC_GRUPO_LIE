# NMPC geométrico + SQP + ADMM com Hessiana SIMÉTRICA do objetivo

Atualização integral do projeto anterior `NMPC_SQP_ADMM_GSYM_GLOBAL`.

## Principais mudanças

1. **Removida toda chamada a `symbolic_Hessian` para a função objetivo.** A nova `symbolic_hessian_simetrica_pullback.m` calcula **apenas** a segunda derivada simétrica do pullback exponencial e o gradiente. Não constrói `A_skew`, comutadores, pseudoinversa ou `L2` não simétrica.
2. Mantida a formulação de `G_sym` do horizonte completo, com dois modos: `global_direct` (direto, 51 nós, potencialmente caro) e `sparse_equivalent` (montagem exata por estágio quando o objetivo é separável).
3. **Padrão do controlador: `sparse_equivalent`**, pois `N=50` pode tornar a diferenciação global simbólica impraticável.
4. **Padrão da exponencial simbólica: `jet2`**. Como gradiente e Hessiana são avaliados em incremento zero, substitui-se `exp(S)` por `I+S+S*S/2`. Isso produz exatamente o mesmo gradiente e a mesma Hessiana em zero para funções `C²` (termos omitidos são de ordem ≥3). Para reprodução literal do método original, passe `'expm'` como último argumento à função geradora; nesse modo, `expm_casadi.m` precisa estar no path.
5. **Mantido o ADMM sem matriz de escalonamento D**: `H_aug = H_i+rho*Ls'*Ls+rho*Le'*Le` para interfaces existentes.
6. **Mantidas as QPs não condensadas**: cada uma possui 76 variáveis, 40 igualdades dinâmicas e 78 desigualdades de estado; os limites dos 8 controles são tratados por `lbx/ubx`.
7. Sem custo terminal. A última cópia compartilhada de cada bloco não recebe custo próprio, evitando contagem dupla; a dinâmica e o consenso permanecem explícitos.

## Arquivos

- `nmpc_multirotor_sqp_admm_Gsym_global.m`: controlador completo, malha fechada.
- `derivadas_globais_horizonte_NMPC.m`: gerador global ou montagem esparsa da Hessiana simétrica.
- `symbolic_hessian_simetrica_pullback.m`: derivadas simétricas do pullback, independente de `symbolic_Hessian`.
- `derivadas_restricoes_NMPC.m`: Jacobianas locais de dinâmica e desigualdades; **continua dependendo do seu `symbolic_Jacobian.m`**, mas não da Hessiana original.
- `project_euler_state_symbolic.m`: auxiliar do modelo.
- `test_hessiana_simetrica_pullback.m`: teste SO(2) e translação (somente CasADi).
- `test_particionamento_global_NMPC.m`: teste de dupla contagem (sem CasADi).
- `test_hessiana_global_simbolica_NMPC.m`: comparação global-direta e local-equivalente para N=2 (requer modelo).

## Executar

```matlab
addpath(pwd,'-begin');
rehash
which derivadas_globais_horizonte_NMPC -all
which symbolic_hessian_simetrica_pullback -all

test_hessiana_simetrica_pullback
 test_particionamento_global_NMPC
% Opcional (demora mais):
% test_hessiana_global_simbolica_NMPC
nmpc_multirotor_sqp_admm_Gsym_global
```

No controlador, a configuração padrão é:

```matlab
hessian_backend = 'sparse_equivalent';
obj_global = derivadas_globais_horizonte_NMPC(N,Q,R,k1,k2,hessian_backend,'jet2');
```

Para forçar a derivação direta de todo o horizonte, altere para `hessian_backend = 'global_direct';`, **com cuidado com o uso de memória**.

## Formulação

As 1420 coordenadas globais são `[dq_0;...;dq_50;du_0;...;du_49]` (dimensão `20×51+8×50`). Para cada estágio, a Hessiana local é 28×28: as outras 20 coordenadas de `q_{k+1}` estavam estruturalmente nulas na Hessiana do objetivo, pois o custo depende só de `(q_k,u_k)`.

No modo `sparse_equivalent`, o código usa um mesmo grupo de 33×33 para estado e controle de um estágio, constrói `H_global` por seleção das 28 coordenadas, e distribui as submatrizes para 25 blocos de 76 variáveis, respeitando as fronteiras compartilhadas.

Na derivação **direta** são mantidas 51 matrizes de 33×33; o bloco de controle do nó terminal é apenas auxiliar e suas 8 coordenadas são descartadas, sem adicionar variável de decisão nem custo. Não há custo terminal.

**Limitações:** não inclui o modelo do multirrotor (`Init`, `Equilibrio`, `IntegrationEuler`, `Attitude_error`), o arquivo original de `symbolic_Jacobian.m`, nem o CasADi/qpOASES. Esses elementos precisam estar no path. Não há validação da simulação completa em MATLAB neste ambiente. Os resultados de otimização dependem de convergência do ADMM (`r`,`s`,`gap`); `ok=1` após rollout de restauração não garante essa convergência.
