"""Build the manuscript's tables and figures from the reported Source Data."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'DO':'DO (separately tuned)','PI_mechanism_legacy':'PI adaptive','DE_raw_concat':'DE raw','PI_static_legacy':'PI static',
 'DO-TEC-only':'DO TEC','PI-TEC-only':'PI TEC projected','DE-ZScore-c05':'DE ZScore','PI-RatioProj':'PI ZScore projected'}
COLORS=['#516C86','#B27252','#497E72','#A08BB4','#87847A']


def tex(s):
    return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')


def table(path, caption, label, header, rows, widths=None, long=False):
    content=' '.join([caption,*header,*[str(v) for row in rows for v in row]])
    terms={'RMSE':'root mean squared error','MAE':'mean absolute error','TECU':'total electron content unit','PI':'physics-inspired','DO':'data-only','DE':'driver-enhanced','SD':'sample standard deviation','SSE':'sum of squared errors','SAE':'sum of absolute errors','GPT-2':'Generative Pre-trained Transformer 2','LoRA':'low-rank adaptation','AMP':'automatic mixed precision','FLOP':'floating-point operation','CODE':'Center for Orbit Determination in Europe'}
    import re
    definitions=[abbr+', '+meaning for abbr,meaning in terms.items() if re.search(r'\b'+abbr+r'\b',content) and meaning.lower() not in caption.lower()]
    if definitions:caption+=' '+ '; '.join(definitions)+'.'
    env='longtable' if long else 'tabular' 
    spec=widths or ('l'+'r'*(len(header)-1))
    start=(r'\begin{longtable}{'+spec+'}\n'+r'\caption{'+caption+r'}\label{'+label+r'}\\'+'\n') if long else (r'\begin{table}[htbp]\centering\small'+'\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+r'\begin{tabular}{'+spec+'}\n')
    s=start+r'\toprule'+'\n'+' & '.join(header)+r'\\\midrule'+'\n'
    if long:s+=r'\endfirsthead\toprule '+' & '.join(header)+r'\\\midrule\endhead'+'\n'
    s+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)
    s+='\n'+r'\bottomrule\end{'+env+'}' + ('' if long else r'\end{table}')+'\n'
    path.write_text(s)


def pm(mean,sd):
    return f'{mean:.3f}'+(r'$\pm$'+f'{sd:.3f}' if np.isfinite(sd) else '')


def driver_table(fs, tabs):
    """Regenerate only Table 4 from the existing five-seed summary."""
    g=fs[fs.dataset=='gim'];c=fs[fs.dataset=='china'];rows=[]
    variants=[('', 'Full'),('__task165__no_solar_index','Without solar'),('__task165__no_geomagnetic','Without geomagnetic'),('__task165__no_external_activity','Without activity')]
    for suffix,label in variants:
        gr=g[g.candidate_id=='gim_de_film_c06'+suffix].iloc[0]
        vals=[label,pm(gr.RMSE_mean,gr.RMSE_sample_sd)]
        for role in ['development_evaluation_2015_H2','retrospective_2019','retrospective_2024']:
            cr=c[(c.candidate_id=='china_de_film_c07'+suffix)&(c.role==role)].iloc[0];vals.append(pm(cr.RMSE_mean,cr.RMSE_sample_sd))
        rows.append(vals)
    table(tabs/'drivers.tex',r'Feature-wise linear modulation (FiLM) input-category removal. RMSE in TECU is mean$\pm$sample SD across five seeds per configuration. Masks apply during training and inference, and all four rows keep solar-zenith geometry as an input. Solar denotes the 10.7-cm solar radio flux (F10.7); geomagnetic denotes the auroral electrojet (AE), disturbance storm time (Dst) and planetary geomagnetic (Kp) indices. Global ionospheric maps (GIM; 2015 and 2019--2022) use 1,826 daily windows. China H2 denotes July--December 2015 (184 windows); China 2019 and 2024 use 365 and 366 windows, respectively.', 'tab:drivers',['Inputs','GIM','China H2','China 2019','China 2024'],rows,'lrrrr')
    return variants


def build(source, out):
    source,out=Path(source),Path(out)
    figs=out/'figures';tabs=out/'tables';figs.mkdir(parents=True,exist_ok=True);tabs.mkdir(parents=True,exist_ok=True)
    read=lambda p:pd.read_csv(source/p)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    def save(fig,name):
        for ext in ['pdf','svg']:fig.savefig(figs/f'{name}.{ext}',bbox_inches='tight')
        preview=out/'build/figures';preview.mkdir(parents=True,exist_ok=True)
        fig.savefig(preview/f'{name}.png',dpi=180,bbox_inches='tight')
        plt.close(fig)
    hist=read('global_historical/model_summary.csv')
    matched=read('global_matched/SEED_EVALUATION.csv')
    fusion=read('fusion/per_run_role_metrics.csv');fs=read('fusion/five_seed_summary.csv')
    rows=[]
    for name in ['DO','PI_mechanism_legacy','DE_raw_concat','ST_GPT_stable']:
        r=hist[hist.model==name].iloc[0]
        rows.append([NAMES.get(name,'TEC-MoLLM'), 'Separately tuned',int(r.n_seeds),pm(r.rmse_tecu_seed_mean,r.rmse_tecu_seed_sample_sd),pm(r.mae_tecu_seed_mean,r.mae_tecu_seed_sample_sd)])
    for name,g in matched.groupby('method',sort=False):
        rows.append([NAMES[name],'Matched / reference',len(g),pm(g.RMSE.mean(),g.RMSE.std()),pm(g.MAE.mean(),g.MAE.std())])
    film=fs[fs.candidate_id=='gim_de_film_c06'].iloc[0]
    rows.append(['DE FiLM','Conditioning',5,pm(film.RMSE_mean,film.RMSE_sample_sd),pm(film.MAE_mean,film.MAE_sample_sd)])
    for folder,name in [('gated','DE Gated'),('crossattn','DE CrossAttn'),('gnntrans','GNN--Transformer')]:
        r=read(folder+'/summary.csv').iloc[0]
        rows.append([name,'Conditioning' if folder!='gnntrans' else 'Adapted',5,pm(r.RMSE_mean,r.RMSE_sample_sd),pm(r.MAE_mean,r.MAE_sample_sd)])
    mixer=read('mixer/metrics_overall_year_lead.csv');mixer=mixer[mixer.dimension=='overall'].iloc[0]
    rows.append(['Mixer--SimVP','External transfer',1,f'{mixer.RMSE:.3f}',f'{mixer.MAE:.3f}'])
    for _, reference in read('baselines/trivial_baselines_summary.csv').iterrows():
        rows.append([reference['model'], 'Reference forecast', '--',
                     f'{reference.RMSE:.3f}', f'{reference.MAE:.3f}'])
    table(tabs/'global.tex',r'Global 24-hour point forecasts in TECU over 1,826 daily windows (2015 and 2019--2022). Values are seed mean$\pm$sample SD. Separately tuned and matched runs have different optimization histories. Matched physical contrasts are PI TEC versus DO TEC and PI ZScore versus the reused DE ZScore reference; the latter uses its original training schedule. Mixer is one external checkpoint. Complete original baseline coverage and per-seed values are in the Supplementary Information. 24-h persistence repeats the maps from the previous day at the same UTC times; C1PG is the CODE one-day predicted GIM product. TEC, total electron content; ZScore, training-set standardization; CODE, Center for Orbit Determination in Europe; GIM, global ionosphere map; UTC, Coordinated Universal Time; FiLM, feature-wise linear modulation.', 'tab:global',['Model','Configuration group','$n$','RMSE','MAE'],rows,'p{.29\linewidth}p{.23\linewidth}rrr')
    regional_table(read, tabs, fs)
    variants=driver_table(fs,tabs)
    # Global primary figure: seed values, matched differences and lead curves.
    fig,ax=plt.subplots(1,3,figsize=(7.2,2.9),gridspec_kw={'width_ratios':[1.15,1,1.1]},layout='constrained')
    order=['DO-TEC-only','PI-TEC-only','DE-ZScore-c05','PI-RatioProj']
    for i,name in enumerate(order):
        vals=matched[matched.method==name].RMSE.to_numpy();ax[0].scatter(i+np.linspace(-.12,.12,len(vals)),vals,s=16,c=COLORS[i],zorder=3);ax[0].plot([i-.22,i+.22],[vals.mean()]*2,c='k',lw=1)
    ax[0].set_xticks(range(4),['DO TEC','PI TEC','DE ZScore','PI ZScore'],rotation=40,ha='right');ax[0].set_ylabel('RMSE (TECU)');ax[0].set_title('a  Training configurations',loc='left',fontweight='bold')
    piv=matched.pivot(index='seed',columns='method',values='RMSE')
    for i,(a,b) in enumerate([('PI-TEC-only','DO-TEC-only'),('PI-RatioProj','DE-ZScore-c05')]):
        vals=piv[a]-piv[b];ax[1].scatter(i+np.linspace(-.13,.13,5),vals,s=18,c=COLORS[i]);ax[1].plot([i-.22,i+.22],[vals.mean()]*2,c='k')
    ax[1].axhline(0,c='.6',lw=.7);ax[1].set_xticks([0,1],['TEC inputs','TEC + drivers'],rotation=40,ha='right');ax[1].set_ylabel('PI minus comparator RMSE (TECU)');ax[1].set_title('b  Paired seeds',loc='left',fontweight='bold')
    leads=read('global_matched/YEAR_LEAD_METRICS.csv');ls=leads.groupby(['method','seed','lead'])[['sse','n_valid']].sum();ls['RMSE']=np.sqrt(ls.sse/ls.n_valid);ls=ls.reset_index()
    for i,name in enumerate(order):
        z=ls[ls.method==name].groupby('lead').RMSE.mean();ax[2].plot(z.index*2,z.values,c=COLORS[i],label=NAMES[name],lw=1.4)
    ax[2].set(xlabel='Forecast lead (h)',ylabel='RMSE (TECU)');ax[2].set_title('c  Lead dependence',loc='left',fontweight='bold');ax[2].legend(fontsize=6.5)
    save(fig,'Figure_2')
    # Regional periods and lead-specific empirical interval coverage.
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained');yr=read('china/year_metrics.csv')
    for i,name in enumerate(['DO','PI_mechanism_legacy','PI_static_legacy']):
        for j,year in enumerate([2015,2019,2024]):
            z=yr[(yr.model==name)&(yr.year==year)].rmse_tecu.to_numpy();x=j+(i-1)*.23;ax[0].scatter(x+np.linspace(-.055,.055,len(z)),z,s=9,c=COLORS[i],alpha=.7);ax[0].errorbar(x,z.mean(),yerr=z.std(ddof=1),fmt='_',c=COLORS[i],capsize=2)
        ax[0].plot([],[],color=COLORS[i],label=NAMES[name])
    ax[0].set_xticks([0,1,2],['2015','2019','2024']);ax[0].set_ylabel('RMSE (TECU)');ax[0].set_title('a  Regional annual errors',loc='left',fontweight='bold');ax[0].legend(fontsize=7)
    cal=read('china/calibration_summary.csv')
    for i,role in enumerate(['development_2015_h2','retrospective_2019','retrospective_2024']):
        for name,style in [('DO','--'),('PI_mechanism_legacy','-')]:
            z=cal[(cal.role==role)&(cal.model==name)&(cal.activity=='All')].sort_values('lead');ax[1].plot(z.lead_hours,z.empirical_coverage_seed_mean*100,style,c=COLORS[i],label=f'{role[-4:] if role.startswith("retrospective") else "2015 H2"} '+('DO' if name=='DO' else 'PI'))
    ax[1].axhline(95,c='.5',ls=':',lw=1);ax[1].set(xlabel='Forecast lead (h)',ylabel='Empirical coverage (%)',ylim=(88,101));ax[1].set_title('b  Calendar calibration',loc='left',fontweight='bold');ax[1].legend(ncol=2,fontsize=6.5)
    save(fig,'Figure_3')
    # Input removal: individual training seeds shown.
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
    for axis,dataset,role,baseid in [(ax[0],'gim','evaluation_2015_2019_2020_2021_2022','gim_de_film_c06'),(ax[1],'china','retrospective_2024','china_de_film_c07')]:
        for i,(suffix,label) in enumerate(variants):
            vals=fusion[(fusion.candidate_id==baseid+suffix)&(fusion.role==role)].RMSE.to_numpy();axis.scatter(i+np.linspace(-.12,.12,5),vals,s=15,c=COLORS[i]);axis.errorbar(i,vals.mean(),yerr=vals.std(ddof=1),fmt='_',c='k',capsize=3)
        axis.set_xticks(range(4),['Full','No solar','No geomagnetic','No activity'],rotation=25,ha='right');axis.set_ylabel('RMSE (TECU)')
    ax[0].set_title('a  Global 2015 and 2019–2022',loc='left',fontweight='bold');ax[1].set_title('b  Regional 2024',loc='left',fontweight='bold');save(fig,'Figure_5')
    # Supplement: preserved adaptive-weight trajectories, not a convergence claim.
    w=read('methods/regional_weight_trajectories.csv');terms=sorted(w.term.unique());fig,axes=plt.subplots(2,4,figsize=(7.2,4.2),layout='constrained')
    for axis,term in zip(axes.flat,terms):
        for _,z in w[w.term==term].groupby('run_id'):axis.plot(z.raw_recorded_epoch,z.log_variance,lw=.7,alpha=.65)
        axis.set(title={'adv':'Gradient alignment (n=10)','chem':'Production–loss (n=10)','diff':'Graph temporal (n=19)','night':'Night growth (n=10)','nonneg':'Nonnegativity (n=19)','rot':'ROT (n=19)','roti':'TSI (n=19)'}[term],xlabel='Recorded epoch',ylabel='Log variance')
    for axis in list(axes.flat)[len(terms):]:axis.axis('off')
    save(fig,'Figure_S5')
    # Valid driver diagnostics kept separate from the removed chemistry ablation.
    dr=read('diagnostics/driver/driver_consistency_runs.csv');cols=[c for c in dr if c.startswith('score_')]
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.4),layout='constrained')
    for axis,col in zip(axes.flat,cols):
        for i,(name,z) in enumerate(dr.groupby('paper_slug',sort=True)):
            vals=z[col].dropna().to_numpy();axis.scatter(i+np.linspace(-.1,.1,len(vals)),vals,s=10,c=COLORS[i]);axis.errorbar(i,vals.mean(),yerr=vals.std(ddof=1),fmt='_',color='k',capsize=2)
        axis.set_xticks([0,1,2],['DE','DO','PI'],rotation=0);axis.set_title({'score_corr_roti_dstn':'TSI and negative Dst','score_corr_absrot_dstn':'Absolute ROT and negative Dst','score_corr_roti_kp':'TSI and Kp','score_corr_absrot_kp':'Absolute ROT and Kp','score_auc_roti':'TSI storm AUC','score_auc_absrot':'Absolute ROT storm AUC'}[col],fontsize=8);axis.set_ylabel('Agreement score')
    save(fig,'Figure_S4')
    # Full-region event diagnostic; points are catalog events, not independent storms.
    # SI long tables of numerical results, using exact saved aggregates.
    rows=[]
    roles=read('china/role_summary.csv')
    for _,r in roles.iterrows():rows.append([NAMES[r.model],tex(r.role.replace('development_','').replace('retrospective_','').replace('_h2',' H2')),pm(r.rmse_tecu_seed_mean,r.rmse_tecu_seed_sample_sd),pm(r.mae_tecu_seed_mean,r.mae_tecu_seed_sample_sd)])
    table(tabs/'regional_roles.tex',r'Regional calendar-role errors in TECU; ten seeds, mean$\pm$sample SD. H2, 2019 and 2024 contain 184, 365 and 366 daily windows.', 'tab:roles_si',['Model','Period','RMSE','MAE'],rows,'llrr',True)
    rows=[]
    for (name,role),z in cal[cal.activity=='All'].groupby(['model','role']):rows.append([NAMES[name],role.replace('retrospective_','').replace('development_','').replace('_h2',' H2').replace('_',r'\_'),f'{100*z.empirical_coverage_seed_mean.min():.2f}--{100*z.empirical_coverage_seed_mean.max():.2f}',f'{z.mean_width_tecu_seed_mean.min():.2f}--{z.mean_width_tecu_seed_mean.max():.2f}'])
    table(tabs/'calibration.tex',r'Ranges across the twelve separately calibrated leads of seed-mean empirical coverage (percent) and interval width (TECU). Thresholds are estimated separately for each model and seed from 181 days. Full lead and activity results, including all 480 thresholds, are in Source Data.', 'tab:calibration',['Model','Period','Coverage','Width'],rows,'llrr',True)
    rows=[]
    for _,r in read('h96/mean_sampleSD.csv').iterrows():
        if r.group in ['old','extension'] and r.selection in ['lead_24h','lead_48h','lead_72h','lead_96h','day_4']:
            rows.append([r.candidate_id.replace('-H96',''),{'old':'primary'}.get(r.group,r.group),tex(r.selection),pm(r.RMSE_mean,r.RMSE_sampleSD),pm(r.MAE_mean,r.MAE_sampleSD)])
    table(tabs/'horizon.tex',r'Direct 96-hour forecasts. RMSE and MAE in TECU are mean$\pm$sample SD over seeds 42, 1024 and 2025. An endpoint is one two-hour lead; day 4 pools leads 37--48. The primary and extension groups contain 1,811 and 1,087 origins.', 'tab:h96',['Method','Group','Selection','RMSE','MAE'],rows,'lllrr',True)
    rows=[]
    for _,r in matched.iterrows():rows.append([NAMES[r.method],int(r.seed),f'{r.RMSE:.6f}',f'{r.MAE:.6f}'])
    table(tabs/'matched_seeds.tex',r'All prescribed optimization seeds in the global matched comparison. RMSE and MAE are pooled over 2015 and 2019--2022 within each seed, in TECU.', 'tab:matched_seeds',['Model','Seed','RMSE','MAE'],rows,'llrr',True)
    restored_materials(read, source, tabs, save)
    return {'main_figures':6,'supplementary_figures':6,'figure1':'PowerPoint schematic'}


def regional_table(read,tabs,fs):
    roles=read('china/role_summary.csv');rows=[]
    for role,frole,label in [('development_2015_h2','development_evaluation_2015_H2','2015 H2'),('retrospective_2019','retrospective_2019','2019'),('retrospective_2024','retrospective_2024','2024')]:
        for model in ['PI_mechanism_legacy','DO','DE_raw_concat']:
            r=roles[(roles.role==role)&(roles.model==model)].iloc[0]
            rows.append([label,NAMES[model],int(r.n_seeds),pm(r.rmse_tecu_seed_mean,r.rmse_tecu_seed_sample_sd),pm(r.mae_tecu_seed_mean,r.mae_tecu_seed_sample_sd)])
        for model,name in [('china_de_zscore_c10','DE ZScore'),('china_de_film_c07','DE FiLM')]:
            r=fs[(fs.role==frole)&(fs.candidate_id==model)].iloc[0]
            rows.append([label,name,int(r.n_seeds),pm(r.RMSE_mean,r.RMSE_sample_sd),pm(r.MAE_mean,r.MAE_sample_sd)])
    table(tabs/'china.tex',r'Regional errors on the same processed product: 184 daily origins in 2015 H2, 365 in 2019 and 366 in 2024, each with twelve leads and 2,911 cells. Values are arithmetic seed mean$\pm$sample SD; $n$ counts training seeds. PI, adaptive physics-inspired; DO, data-only; DE, driver-enhanced; ZScore, standardized concatenation; FiLM, feature-wise linear modulation. RMSE and MAE are in TECU. Adaptive PI/DO/raw DE and later conditioning runs have different training histories. Figure 3a instead shows full-year 2015.', 'tab:china',['Period','Model','$n$','RMSE','MAE'],rows)
    # Preserve the former main pooled/static table in the SI.
    rows=[]
    for _,r in read('china/model_summary.csv').iterrows():
        rows.append([NAMES[r.model],'2015' if r.split=='test' else '2019 + 2024',int(r.n_seeds),pm(r.rmse_tecu_seed_mean,r.rmse_tecu_seed_sample_sd),pm(r.mae_tecu_seed_mean,r.mae_tecu_seed_sample_sd)])
    table(tabs/'china_pooled.tex',r'Full-year 2015 and pooled 2019+2024 regional errors of the separately tuned models. RMSE and MAE in TECU, arithmetic seed mean$\pm$sample SD. The periods have 365 and 731 origins, respectively; $n$ counts seeds. PI static is reported alongside adaptive PI, DO and raw DE.', 'tab:china_pooled',['Model','Period','$n$','RMSE','MAE'],rows,long=True)


def restored_materials(read,source,tabs,save):
    from lib.paper_statistics import paired_pooled_rmse_interval
    saved=json.loads((source/'global_matched/FINAL_ANALYSIS.json').read_text())['comparisons']
    for file,a,b,key in [('TEC_PI_DO','PI-TEC-only','DO-TEC-only','TEC-Phys_minus_TEC-Sup'),('DRIVER_PI_DE','arm_a','zscore','TEC+Driver-Phys_minus_TEC+Driver-Sup')]:
        interval=paired_pooled_rmse_interval(read(f'global_matched/BLOCK_STATISTICS_{file}.csv'),a,b)
        np.testing.assert_allclose([interval['mean_pooled_rmse_delta'],interval['low'],interval['high']], [saved[key]['mean'],*saved[key]['pooled_RMSE_14day_CI']['interval']],atol=1e-12,rtol=0)
        print(file,interval)
    # H96 full curves and endpoints use group rows only, never their annual duplicates.
    h=read('h96/per_seed_year_group_metrics.csv');h['RMSE']=np.sqrt(h.SSE/h.n);h['MAE']=h.SAE/h.n
    methods=['DO-TEC-only-H96','PI-TEC-only-H96','DE-ZScore-H96']
    names=['DO TEC','PI TEC projected','DE ZScore']
    def curves(axis,group):
        for i,name in enumerate(methods):
            z=h[(h.group==group)&(h.candidate_id==name)&h.selection.str.startswith('lead_')].copy();z['hours']=z.selection.str.extract(r'lead_(\d+)h').astype(int)
            for _,t in z.groupby('seed'):
                t=t.sort_values('hours');axis.plot(t.hours,t.RMSE,c=COLORS[i],lw=.55,alpha=.45)
            t=z.groupby('hours').RMSE.mean();axis.plot(t.index,t,c=COLORS[i],lw=1.6,label=names[i])
        axis.set(xlabel='Forecast lead (h)',ylabel='RMSE (TECU)',xticks=[2,24,48,72,96]);axis.legend(fontsize=7)
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained');curves(ax[0],'old')
    ax[0].set_title('a  2015 and 2019–2022',loc='left',fontweight='bold')
    for j,hour in enumerate([48,72,96]):
        z=h[(h.group=='old')&(h.selection==f'lead_{hour}h')].pivot(index='seed',columns='candidate_id',values='RMSE');d=z[methods[1]]-z[methods[0]]
        ax[1].scatter(j+np.linspace(-.1,.1,3),d,s=25,c=COLORS[1]);ax[1].plot([j-.2,j+.2],[d.mean()]*2,c='k')
        ax[0].axvline(hour,c='.7',lw=.5,ls=':')
        print('H96 endpoint',hour,'delta',d.mean(),d.std(),'improved',int((d<0).sum()))
    ax[1].axhline(0,c='.5',lw=.8);ax[1].set(xticks=[0,1,2],xticklabels=['48','72','96'],xlabel='Forecast lead (h)',ylabel='PI minus DO RMSE (TECU)');ax[1].set_title('b  Paired endpoint differences',loc='left',fontweight='bold');save(fig,'Figure_6')
    fig,ax=plt.subplots(figsize=(5.3,3),layout='constrained');curves(ax,'extension');ax.set_title('2023–2025 extension',loc='left',fontweight='bold');save(fig,'Figure_S6')
    rows=[]
    for hour in [48,72,96]:
        for method,name in zip(methods,names):
            z=h[(h.group=='old')&(h.selection==f'lead_{hour}h')&(h.candidate_id==method)]
            rows.append([hour,name,pm(z.RMSE.mean(),z.RMSE.std()),pm(z.MAE.mean(),z.MAE.std())])
    table(tabs/'horizon_endpoints.tex',r'Original-year H96 endpoints: 2015 and 2019--2022, 1,811 daily origins and seeds 42, 1024 and 2025. RMSE and MAE in TECU are arithmetic mean$\pm$sample SD after pooling SSE/SAE/count within seed. Each endpoint is one lead of the same selected forty-eight-output model, not an independent experiment. DO, data-only; PI, projected physics-inspired; DE, standardized driver-enhanced.', 'tab:h96_endpoints',['Lead (h)','Model','RMSE','MAE'],rows)
    # June cell series and full-region catalogue events.
    case=json.loads((source/'events/figure5_case_selected.json').read_text());ts=read('events/figure5_timeseries_seed42.csv')
    ids=case['paper_slugs'];labels=['PI adaptive','DO','DE raw'];styles=['-','--',':']
    identity=read('methods/regional_training_identity.csv')
    assert all(Path(case['run_dirs'][slug]).name.split('-')[-1] in set(identity.run_id) for slug in ids)
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.1),layout='constrained')
    for axis,station,title in zip(axes[:2],['beijing','haikou'],['a  Beijing grid cell','b  Haikou grid cell']):
        z=ts[ts.station==station].sort_values('time_unix');hours=(z.time_unix-z.time_unix.min())/3600
        axis.plot(hours,z.tec_truth,c='k',lw=1.6,label='Processed target')
        errors={}
        for i,(slug,label,style) in enumerate(zip(ids,labels,styles)):
            val=z['tec_'+slug];rmse=float(np.sqrt(np.mean((val-z.tec_truth)**2)));errors[label]=rmse
            axis.plot(hours,val,style,c=COLORS[[1,0,2][i]],lw=1.15,label=f'{label} ({rmse:.3f})')
        axis.axvline(24,c='.5',ls='--',lw=.7);axis.set(xticks=[0,12,24,36,48],xlabel='Hours from 22 June 00 UTC',ylabel='TEC (TECU)');axis.set_title(title,loc='left',fontweight='bold');axis.legend(fontsize=5.9);print(station,errors)
    ev=read('events/event_full_domain_summary.csv');pair=ev.pivot(index=['role','event_id'],columns='model',values='rmse_tecu_seed_mean')
    role_names={'calibration_2015_h1':'2015 H1 calibration','development_evaluation_2015_h2':'2015 H2','evaluation_2015_h2':'2015 H2','robustness_2019':'2019','robustness_2024':'2024'}
    for i,(role,z) in enumerate(pair.groupby(level=0)):axes[2].scatter(z.DO,z.PI_mechanism_legacy,s=10,c=COLORS[i],label=role_names.get(role,role),alpha=.7)
    lim=max(pair.DO.max(),pair.PI_mechanism_legacy.max())*1.05;axes[2].plot([0,lim],[0,lim],c='.5',lw=.8);axes[2].set(xlabel='DO RMSE (TECU)',ylabel='PI RMSE (TECU)');axes[2].set_title('c  54 regional events',loc='left',fontweight='bold');axes[2].legend(fontsize=5.9);save(fig,'Figure_4')
    snap=np.load(source/'events/figure5_snapshots_seed42.npz');assert str(snap['time_utc'])=='2015-06-23T04:00:00Z'
    fields=[snap['truth']]+[snap[k]-snap['truth'] for k in ids];vmax=max(abs(z).max() for z in fields[1:])
    fig,ax=plt.subplots(2,2,figsize=(7.2,5.3),layout='constrained')
    for i,(axis,field,title) in enumerate(zip(ax.flat,fields,['a  Processed target','b  PI minus target','c  DO minus target','d  Raw DE minus target'])):
        im=axis.pcolormesh(snap['lon_grid'],snap['lat_grid'],field,shading='nearest',cmap='viridis' if i==0 else 'RdBu_r',vmin=0 if i==0 else -vmax,vmax=field.max() if i==0 else vmax,rasterized=True)
        axis.scatter([116,110],[40,20],s=16,marker='x',c='k',linewidths=.8);axis.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)',xlim=(70,140),ylim=(15,55));axis.set_title(title,loc='left',fontweight='bold')
        if i==0:fig.colorbar(im,ax=axis,label='TEC (TECU)',shrink=.85)
        if i==3:fig.colorbar(im,ax=[ax[0,1],ax[1,0],ax[1,1]],label='Prediction minus target (TECU)',shrink=.85)
    save(fig,'Figure_S3')
    # Historical lead and latitude diagnostics, preserving actual cohort sizes.
    lead=read('global_historical/lead_metrics.csv');lat=read('global_historical/latitude_band_metrics.csv')
    histnames=['DO','PI_mechanism_legacy','DE_raw_concat','ST_GPT_stable'];display=['DO (separately tuned)','PI adaptive','DE raw','TEC-MoLLM']
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
    bands=['abs_lat_0_5','abs_lat_5_20','abs_lat_20_40','abs_lat_40_60','abs_lat_60_90']
    for i,(name,label) in enumerate(zip(histnames,display)):
        z=lead[lead.model==name];n=z.seed.nunique()
        for _,g in z.groupby('seed'):ax[0].plot(g.lead_hours,g.rmse_tecu,c=COLORS[i],lw=.45,alpha=.3)
        g=z.groupby('lead_hours').rmse_tecu.mean();ax[0].plot(g.index,g,c=COLORS[i],label=f'{label} (n={n})')
        for j,band in enumerate(bands):
            g=lat[(lat.model==name)&(lat.geographic_latitude_band==band)];v=g.rmse_tecu.to_numpy();x=j+(i-1.5)*.17
            ax[1].scatter(x+np.linspace(-.035,.035,len(v)),v,c=COLORS[i],s=8,alpha=.7);ax[1].errorbar(x,v.mean(),yerr=v.std(ddof=1),fmt='_',c=COLORS[i],capsize=2)
    ax[0].set(xlabel='Forecast lead (h)',ylabel='RMSE (TECU)');ax[0].set_title('a  Original-year lead errors',loc='left',fontweight='bold');ax[0].legend(fontsize=6.5)
    ax[1].set(xticks=range(5),xticklabels=['0–5','5–20','20–40','40–60','60–90'],xlabel='Absolute geographic latitude (°)',ylabel='RMSE (TECU)');ax[1].set_title('b  Within-seed latitude pooling',loc='left',fontweight='bold');save(fig,'Figure_S1')
    monthly=read('global_historical/figureS1_monthly_rmse.csv');amp=read('global_historical/figureS2_amplitude_bins.csv')
    fig,ax=plt.subplots(1,3,figsize=(7.2,3),layout='constrained')
    slugs=['do_gwn','a1_pinn_confirm_pi_pinn_drvonly_adv_lp0p10','st_gpt_stable','iri_2020']
    for i,(slug,label) in enumerate(zip(slugs,['DO','PI adaptive','TEC-MoLLM','IRI-2020'])):
        z=monthly[monthly.paper_slug==slug]
        if slug=='iri_2020':
            assert z.groupby('month').rmse.nunique().max()==1;z=z.drop_duplicates('month')
        for _,g in z.groupby('seed'):ax[0].plot(g.month,g.rmse,c=COLORS[i],lw=.55,alpha=.5)
        g=z.groupby('month').rmse.mean();ax[0].plot(g.index,g,c=COLORS[i],lw=1.5,label=label)
    for axis,metric in zip(ax[1:],['rmse','mae']):
        for i,slug in enumerate(slugs[:2]):
            z=amp[amp.paper_slug==slug]
            for _,g in z.groupby('seed'):
                g=g.sort_values('bin_lo');axis.plot(range(4),g[metric],'-o',ms=2,c=COLORS[i],lw=.5,alpha=.55)
            g=z.groupby('bin_lo')[metric].mean();axis.plot(range(4),g,c=COLORS[i],lw=1.5)
        axis.set(xticks=range(4),xticklabels=['0–20','20–40','40–60','≥60'],xlabel='Target TEC bin (TECU)',ylabel=metric.upper()+' (TECU)')
    ax[0].set(xlabel='Calendar month',ylabel='RMSE (TECU)',xticks=[1,3,6,9,12]);ax[0].legend(fontsize=6.5)
    for axis,title in zip(ax,['a  Monthly errors','b  Amplitude RMSE','c  Amplitude MAE']):axis.set_title(title,loc='left',fontweight='bold')
    save(fig,'Figure_S2')
    # Original diagnostic interval is a normal approximation; preserve this identity.
    d=read('diagnostics/driver/driver_consistency_pi_vs_de_pvalues.csv');p=d.p_value_signflip_two_sided.to_numpy();order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    labels=['TSI versus -Dst','Abs. ROT versus -Dst','TSI versus Kp','Abs. ROT versus Kp','TSI storm AUC','Abs. ROT storm AUC'];rows=[]
    for i,(_,r) in enumerate(d.iterrows()):
        se=r.pi_minus_de_std/np.sqrt(r.n_pairs);rows.append([labels[i],f'{r.pi_minus_de_mean:.4f}',f'[{r.pi_minus_de_mean-1.96*se:.4f}, {r.pi_minus_de_mean+1.96*se:.4f}]',f'{p[i]:.4f}',f'{q[i]:.4f}'])
    table(tabs/'driver_paired.tex',r'Paired regional driver-agreement scores, PI minus raw DE, ten seed pairs. Higher scores (toward zero) are better. Intervals are 95\% normal approximations, mean$\pm1.96$ sample SD$/\sqrt{10}$; $p$ is the reported two-sided sign-flip test, and $q$ is Benjamini--Hochberg across these six scores. ROT, rate of TEC; TSI, coarse temporal variability; AUC, area under the receiver operating characteristic curve. Scores are dimensionless and conditional on the selected product.', 'tab:driver_paired',['Score','Difference','95\% interval','$p$','$q$'],rows,long=True)
    budget=read('methods/transformer_budget.csv');rows=[]
    for name,z in budget[budget.row_type=='seed'].groupby('config',sort=False):
        label={'orig-budget':'Original budget','matched-budget elastic + GC':'Elastic control'}.get(name,name)
        rows.append([tex(label),len(z),pm(z.rmse.mean(),z.rmse.std()),pm(z.mae.mean(),z.mae.std())])
    table(tabs/'transformer_budget.tex',r'TEC-MoLLM budget sensitivity, seeds 42, 1024 and 2025. Errors are in TECU, mean$\pm$sample SD, from the reported scalar results. Original budget uses batch 17, accumulation six and node chunk 1,024; the elastic control uses batch 32, accumulation six, chunk 512 and gradient checkpointing. Both use three GPT-2-large blocks, LoRA, twelve input/output maps, Adam, learning rate $3\times10^{-4}$, zero decay, cosine scheduling, AMP, maximum 200 epochs and stopping patience 20. This is not an equal-FLOP or equal-wall-time comparison. The original-budget scalar mean 4.1333 is distinct from the 4.134 re-evaluated prediction cohort.', 'tab:transformer_budget',['Configuration','$n$','RMSE','MAE'],rows,'p{.49\linewidth}rrr',True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--revised-sources',required=True)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--figures-only',action='store_true',help='Draw the ten approved numerical figures; never generate tables or tests.')
    p.add_argument('--only',help='Final figure numbers, for example 2,4,S3 (requires --figures-only).')
    p.add_argument('--legacy-root',help='Fallback root for missing approved small historical inputs.')
    a=p.parse_args()
    if a.figures_only:
        from revised_figures import build_figures
        print(build_figures(a.revised_sources,a.out_dir,a.only,a.legacy_root))
    elif a.only or a.legacy_root:
        p.error('--only and --legacy-root require --figures-only')
    else:
        print(build(a.revised_sources,a.out_dir))


if __name__=='__main__':main()
