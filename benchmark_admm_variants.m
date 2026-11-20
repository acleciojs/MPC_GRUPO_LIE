function table_out = benchmark_admm_variants()
% Comparar, no MESMO modelo e estados iniciais, tres estrategias.
% Requer nmpc_multirotor_sqp_admm_fast.m e dependencias originais.
% ATENCAO: nao e comparacao de garantia matematica.
cfg=struct('T_sim',25,'N',40,'M',8,'max_sqp',1,'sqp_init_steps',0);
names={'ADMM classico','Sobre-relaxado','Nesterov restart'};
a=[1.0 1.6 1.6];
nest=[false false true];
adapt=[true true false]; % isola momentum com rho fixo
n=numel(names);
mean_time=nan(n,1); total_qp=nan(n,1); avg_admm=nan(n,1);
final_err=nan(n,1); accepted=nan(n,1); solved=nan(n,1);
for k=1:n
    fprintf('\n===== BENCHMARK %s =====\n',names{k});
    cfg.alpha=a(k);
    cfg.nesterov=nest(k);
    cfg.adapt_rho=adapt(k);
    r=nmpc_multirotor_sqp_admm_fast(cfg);
    mean_time(k)=mean(r.time);
    total_qp(k)=r.total_qp_solves;
    avg_admm(k)=mean(r.admm);
    final_err(k)=r.err(end);
    accepted(k)=mean(r.alpha>0);
    solved(k)=mean(r.converged);
end
table_out=table(names(:),mean_time,total_qp,avg_admm,final_err,accepted,solved, ...
    'VariableNames',{'Metodo','TempoMedio','TotalQPs','MediaADMM', ...
                     'ErroPosFinal','PassosAceitos','ADMMConvergiu'});
disp(table_out);
end
