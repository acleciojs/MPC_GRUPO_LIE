# SQP geométrico + ADMM rápido: instruções e limites

Esta versão altera **o código 30 estados/8 entradas com decomposição temporal**, não o controlador AKC-NMPC translacional criado anteriormente.

## Executar

Coloque `nmpc_multirotor_sqp_admm_fast.m` no mesmo ambiente dos arquivos externos de sua implementação original (`Init`, `Equilibrio`, `IntegrationEuler`, `Attitude_error`, `project_euler_state_symbolic`, `derivadas_globais_horizonte_NMPC`, `derivadas_restricoes_NMPC`, CasADi e plugin `qpoases`). Ajuste o caminho do CasADi no começo do arquivo, se necessário.

```matlab
r = nmpc_multirotor_sqp_admm_fast(struct('T_sim', 20));
```

Para um experimento mais longo:

```matlab
r = nmpc_multirotor_sqp_admm_fast(struct('T_sim', 1000));
```

Para testar orçamento diferente sem editar o arquivo:

```matlab
r = nmpc_multirotor_sqp_admm_fast(struct( ...
    'T_sim', 50, 'N', 40, 'M', 8, ...
    'alpha', 1.6, 'admm_budgets', [3 5 10], ...
    'nesterov', false));
```

Para comparar estratégias (atenção ao tempo computacional):

```matlab
comparacao = benchmark_admm_variants();
```

## Mudanças principais

- **Oito blocos**, 7 interfaces de consenso, em vez de 20 blocos/19 interfaces para N=40. QPs locais ficam maiores; o ganho líquido de tempo deve ser medido.
- **SQP RTI**: uma SQP na maioria das amostras; até duas nas três primeiras por padrão.
- **ADMM progressivo**: 3 iterações inicialmente; se a direção não passar no teste de redução do custo não linear, mais 5 e depois mais 10 no **mesmo QP**. Limite = 18 iter./SQP.
- **Sobre-relaxação ADMM** de 1.6 na atualização de consenso e dual. Não confundir com passo alpha da SQP.
- **Rho adaptativo** verificado de duas em duas iterações; antes o critério de 5 era inalcançável com apenas 3 iterações.
- **Warm-start dual** deslocado/interpolado e amortecido entre as amostras. Dentro do mesmo QP, a solução do ADMM é preservada ao ampliar o orçamento.
- **Trust region nos incrementos de controle** para impedir passos enormes a partir de linearizações locais.
- **Rollout da dinâmica não linear** usado diretamente quando o ADMM ainda não atingiu consenso. A retração exponencial de toda a trajetória é testada somente quando faz sentido.
- **Custo da trajetória factível** não exige recalcular os defeitos de dinâmica após rollout, reduzindo chamadas ao integrador.
- **Verificação de limites** das trajetórias candidatas; se o ponto nominal for inviável, apenas trajetórias que reduzam a violação são aceitas como *restauração*, sem alegar segurança.
- Estado inicial fixo não recebe restrição linearizada impossível na primeira QP.

## Interpretar a saída

O console mostra `r/eps` e `s/eps`, as razões entre os resíduos primal e dual do ADMM e suas tolerâncias. **Ambos devem ser ≤ 1 para declarar convergência do ADMM.** Isso é diferente de aceitar uma direção inexata para redução do custo não linear do MPC.

`modo=restaurado` significa que um incremento de controle da QP foi aceito após simular a planta não linear sobre todo o horizonte; **não** significa que a QP convergiu. `modo=gradiente-restaurado` indica fallback de descida por adjunto. `modo=none` indica ausência de passo aceito.

Observe `results.err` junto com velocidade, entradas, saturações, custo `results.merit`, `results.alpha`, `results.time`, `results.r_norm`, `results.s_norm`, `results.converged`. Uma pequena redução do custo em um único horizonte não garante convergência do MPC em malha fechada.

## Nesterov

Nesterov está implementado como extrapolação do consenso e dos duais **com reinício por crescimento do resíduo**, mas fica **desligado por padrão**. A sobre-relaxação é a configuração preferida na primeira rodada. Se for testar momentum, mantenha o `rho` fixo inicialmente para não misturar dois mecanismos adaptativos:

```matlab
r = nmpc_multirotor_sqp_admm_fast(struct( ...
   'T_sim', 30, 'nesterov', true, 'adapt_rho', false));
```

A opção experimental com Nesterov não herda automaticamente as garantias do ADMM clássico em todos os QPs locais e não garante menor número de iterações.

## Diagnóstico de não convergência

Se `r_norm` alto e `s_norm` baixo: desacordo entre blocos, tentar aumentar orçamento ADMM ou ajustar rho. Se `r_norm` baixo e `s_norm` alto: consenso está mudando, continuar ADMM ou diminuir rho. Se ambos baixos e `modo=none`: o problema é o passo SQP/globalização ou as restrições, não apenas o ADMM.

Se a posição não converge mesmo com direções aceitas: 0.4 segundo de horizonte (`N=40`, `dt=0.01`), sem custo terminal, pode ser insuficiente para a dinâmica completa. Testar um horizonte maior (por exemplo `N=60`, `M=10`) **após validar** a configuração inicial, reconhecendo custo maior.

Os obstáculos no código são apenas desenhados e NÃO entram nas restrições do NMPC.

## Limitações e transparência

Não houve execução de MATLAB/CasADi neste ambiente porque os arquivos auxiliares específicos do modelo e o MATLAB não estão disponíveis. O arquivo completo foi revisado estaticamente; a convergência real e o tempo dependem do seu modelo e instalação. A dinâmica é a mesma `IntegrationEuler` do original, com projeção posterior no grupo; essa projeção precisa ser consistente com as Jacobianas intrínsecas externas. A Hessiana usada continua sendo a Hessiana do **objetivo**, conforme o código fornecido, e não a Hessiana da Lagrangiana completa.

## Duas checagens novas no início

O código imprime `||f(x_ref,u_hover)-x_ref||_inf`. Se esse valor for grande, `Equilibrio` não corresponde a um ponto fixo da dinâmica discreta utilizada — uma falha que o ajuste do ADMM não resolverá. Também sinaliza defeitos iniciais `h0` maiores que `1e-5` no rollout, que podem revelar incompatibilidade entre `derivadas_restricoes_NMPC` e a projeção/retração adotada.
