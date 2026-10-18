"""Approved numerical figures, drawn at the clean manuscript's physical widths.

No training, inference, new inferential statistics, or ordinary table generation.
All file paths are relative to the canonical Source Data directory. The visual
references informed layout only; this module does not vendor third-party code.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.collections import Collection
from matplotlib.colors import to_rgb, to_rgba_array
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageOps

# main: sn-jnl's 31 TeX picas; SI: A4 less two 23-mm margins.
MAIN_WIDTH = 31 * 12 / 72.27
SI_WIDTH = 164 / 25.4
C = dict(do='#426A86', pi='#B17A41', de='#427B69', projected='#8B78A5',
         reference='#20262C', static='#6B6B6B', raw='#738E7A', transformer='#7D728C')
STYLE = {
 'DO-TEC-only': ('DO TEC', 'do', 'o', '--'),
 'PI-TEC-only': ('PI TEC projected', 'pi', 's', '-'),
 'DE-ZScore-c05': ('DE ZScore', 'de', '^', '-.'),
 'PI-RatioProj': ('PI ZScore projected', 'projected', 'D', ':'),
 'DO': ('DO', 'do', 'o', '--'),
 'PI_mechanism_legacy': ('PI adaptive', 'pi', 's', '-'),
 'DE_raw_concat': ('Raw DE', 'raw', '^', '-.'),
 'PI_static_legacy': ('PI static', 'static', 'D', ':'),
 'ST_GPT_stable': ('TEC-MoLLM', 'transformer', 'D', ':'),
}
# Printed sizes (pt): individual seeds thin, means thicker, guides thin; journal floor 0.3 pt.
SEED_LW, MEAN_LW, MS, MEW, REF_LW = .35, .75, 2.3, .3, .5
# Labelled leads: quarter days on 24-hour axes, whole days on 96-hour axes.
LEAD24, LEAD96 = [6, 12, 18, 24], [24, 48, 72, 96]
METHODS = ['DO-TEC-only', 'PI-TEC-only', 'DE-ZScore-c05', 'PI-RatioProj']
HISTORICAL = ['DO', 'PI_mechanism_legacy', 'DE_raw_concat', 'ST_GPT_stable']
VARIANTS = [('', 'Full'), ('__task165__no_solar_index', 'Without solar'),
            ('__task165__no_geomagnetic', 'Without geomagnetic'),
            ('__task165__no_external_activity', 'Without activity')]
LEGACY_FILES = {
 'global_historical/figureS1_monthly_rmse.csv': 'paper_artifacts_freeze_20260116_r6/figures/figureS1_monthly_rmse.csv',
 'global_historical/figureS2_amplitude_bins.csv': 'paper_artifacts_freeze_20260116_r6/figures/figureS2_amplitude_bins.csv',
 **{f'events/{n}': f'paper_artifacts_revision_r4_20260520_t140/derived/fig03_storm_case/figures/{n}' for n in
    ['figure5_case_selected.json','figure5_timeseries_seed42.csv','figure5_snapshots_seed42.npz']},
}


def setup():
    available = {f.name for f in font_manager.fontManager.ttflist}
    # Metric-compatible fonts only: wider fallbacks such as DejaVu Sans break the S1 and Figure 6 layouts.
    font = next((n for n in ['Arial', 'Helvetica', 'Liberation Sans'] if n in available), None)
    if font is None:
        raise RuntimeError('Figures need Arial, Helvetica or Liberation Sans (e.g. the fonts-liberation package).')
    plt.rcParams.update({
      'font.family': 'sans-serif', 'font.sans-serif': [font], 'font.size': 8,
      'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 8,
      'ytick.labelsize': 8, 'legend.fontsize': 8, 'legend.frameon': False,
      'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .5,
      'xtick.major.width': .5, 'ytick.major.width': .5,
      'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
      'lines.linewidth': MEAN_LW, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
      'savefig.facecolor': 'white', 'savefig.bbox': None,
      'figure.constrained_layout.w_pad': .035, 'figure.constrained_layout.h_pad': .055,
      'figure.constrained_layout.wspace': .08, 'figure.constrained_layout.hspace': .08,
    })
    return font


def panel(ax, letter, title):
    ax.set_title(title, loc='left', pad=7)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-25, 7),
                textcoords='offset points', fontsize=10, fontweight='bold',
                va='bottom', ha='left', annotation_clip=False)


def tint(color, f=.4):
    """Opaque blend toward white: light but not faint, and overlaps never darken."""
    r=np.array(to_rgb(color));return tuple(r+(1-r)*f)


def mean_z(ls):
    """Solid means below broken ones, so dash gaps reveal a coincident solid curve."""
    return 4 if ls=='-' else 4.2


def handle(method, label=None):
    # Legend symbols slightly larger than plotted ones so the shapes stay identifiable.
    name, color, marker, ls = STYLE[method]
    return Line2D([], [], color=C[color], marker=marker, ms=MS+.7, lw=MEAN_LW, linestyle=ls,
                  markerfacecolor='white' if color=='static' else C[color],
                  markeredgecolor=C[color] if color=='static' else 'white',
                  markeredgewidth=.5 if color=='static' else MEW, label=label or name)


def unique(df, keys):
    if df.duplicated(keys).any():
        raise ValueError(f'Non-unique scientific keys {keys}')


def seeds(df, count):
    assert len(df) == count and df.seed.nunique() == count, (len(df), df.seed.nunique(), count)
    return df.sort_values('seed')


def points(ax, x, values, method, sd=False, spread=.10, mean_half=.21, sd_gap=.05):
    _, key, marker, _ = STYLE[method]
    values=np.asarray(values); assert np.isfinite(values).all()
    hollow=key=='static'
    ax.scatter(x+np.linspace(-spread,spread,len(values)),values,s=8,
               marker=marker,facecolor='white' if hollow else C[key],
               edgecolor=C[key] if hollow else 'white',linewidth=.6 if hollow else .4,zorder=3)
    if sd:
        # Mean and sample SD beside, not through, the seed points.
        ax.errorbar(x+spread+sd_gap,values.mean(),yerr=values.std(ddof=1),fmt='_',ms=5,mew=MEAN_LW,
                    capsize=1.8,capthick=.6,elinewidth=.6,color=C[key],zorder=4)
    else:
        ax.plot([x-mean_half,x+mean_half],[values.mean()]*2,color=C['reference'],lw=MEAN_LW,zorder=4)


def curves(ax, df, method, x, y, n, markers=None):
    name,key,marker,ls=STYLE[method]
    unique(df,['seed',x]);assert df.seed.nunique()==n
    expected=sorted(df[x].unique())
    for _,z in df.groupby('seed',sort=True):
        z=z.sort_values(x);assert list(z[x])==expected
        ax.plot(z[x],z[y],color=tint(C[key]),ls='-',lw=SEED_LW,zorder=1)
    z=df.groupby(x,sort=True)[y].mean()
    # markers may be (start, step) so coincident means put their symbols at different leads.
    ax.plot(z.index,z.values,color=C[key],ls=ls,lw=MEAN_LW,marker=marker,ms=MS,mec='white',mew=MEW,
            markevery=markers if markers is not None else 3,label=name,zorder=mean_z(ls))


def paired_lead_delta(df, col, a, b, x, y, n):
    """Seed-paired a minus b at each lead; rows are already pooled within seed and lead."""
    w={}
    for m in (a,b):
        z=df[df[col]==m];unique(z,['seed',x]);w[m]=z.pivot(index='seed',columns=x,values=y)
    d=w[a]-w[b]
    assert len(d)==n and list(w[a].index)==list(w[b].index) and list(w[a].columns)==list(w[b].columns),(a,b,d.shape)
    assert d.notna().all().all() and np.isfinite(d.values).all(),(a,b)
    print('Paired lead mean',a,'-',b,{int(k):round(float(v),4) for k,v in d.mean().items()},flush=True)
    return d


def delta_curves(ax, d, method, markers=3):
    """Every seed difference (thin) and their arithmetic mean (thick); no band."""
    _,key,marker,ls=STYLE[method]
    for _,row in d.iterrows():ax.plot(d.columns,row.values,color=tint(C[key]),lw=SEED_LW,zorder=1)
    m=d.mean();ax.plot(m.index,m.values,color=C[key],ls=ls,lw=MEAN_LW,marker=marker,ms=MS,mec='white',mew=MEW,markevery=markers,zorder=4)
    ax.axhline(0,color='.45',lw=REF_LW,zorder=0)
    return float(np.abs(d.values).max())


def facet_label(ax, text, top=False):
    """Name in a facet's lower-right (or upper-right) corner, which the plotted data leave empty."""
    ax.text(.97,.97 if top else .03,text,transform=ax.transAxes,ha='right',va='top' if top else 'bottom',
            linespacing=1.3,color=C['reference'],zorder=5)


def column_header(ax, entry):
    """A one-entry legend centred above a facet, clear of the panel title line."""
    ax.legend(handles=[entry],loc='lower center',bbox_to_anchor=(.5,1),borderaxespad=2.3)


def label_strip(fig, spec, text):
    """Shared axis label under a row of facets, as an empty axes so the layout reserves its space."""
    ax=fig.add_subplot(spec);ax.axis('off');ax.text(.5,.5,text,ha='center',va='center',transform=ax.transAxes)


class Figures:
    def __init__(self,source):self.source=Path(source)
    def read(self,path):return pd.read_csv(self.source/path)

    def figure_2(self):
        df=self.read('global_matched/SEED_EVALUATION.csv');unique(df,['method','seed'])
        np.testing.assert_allclose(df.RMSE,np.sqrt(df.sse/df.n_valid),rtol=1e-12)
        # Height leaves the float page its original margin with the one-line-longer legend.
        fig=plt.figure(figsize=(MAIN_WIDTH,5.06),layout='constrained')
        # One four-column grid: a,b,d,e span two columns; c has one configuration per column on a shared
        # scale, so the TEC-only pair sits above d and the driver pair above e with aligned edges.
        gs=fig.add_gridspec(4,4,height_ratios=[1,1,.12,.9]);a=fig.add_subplot(gs[0,:2]);b=fig.add_subplot(gs[0,2:])
        cax=[fig.add_subplot(gs[1,0])];cax+=[fig.add_subplot(gs[1,k],sharey=cax[0]) for k in range(1,4)]
        dax=fig.add_subplot(gs[3,:2]);eax=fig.add_subplot(gs[3,2:],sharey=dax)
        for i,m in enumerate(METHODS):points(a,i,seeds(df[df.method==m],5).RMSE,m,spread=.23)
        a.set(xticks=range(4),xticklabels=['DO\nTEC','PI TEC\nprojected','DE\nZScore','PI ZScore\nprojected'],ylabel='RMSE (TECU)',xlim=(-.5,3.5))
        panel(a,'a','Training configurations')
        p=df.pivot(index='seed',columns='method',values='RMSE')
        for i,(m,ref) in enumerate([(METHODS[1],METHODS[0]),(METHODS[3],METHODS[2])]):
            d=p[m]-p[ref];assert len(d)==5 and d.notna().all();points(b,i,d,m)
            print('Figure 2 paired',m,dict(zip(p.index,d)), 'mean',d.mean())
        b.axhline(0,color='.5',lw=REF_LW);b.set(xticks=[0,1],xticklabels=['TEC-only','TEC + drivers'],ylabel='ΔRMSE (TECU)',xlim=(-.55,1.55))
        panel(b,'b','Paired seeds: PI − comparator')
        raw=self.read('global_matched/YEAR_LEAD_METRICS.csv');unique(raw,['method','seed','year','lead'])
        z=raw.groupby(['method','seed','lead'],as_index=False)[['sse','n_valid']].sum();z['RMSE']=np.sqrt(z.sse/z.n_valid);z['hours']=z.lead*2
        assert sorted(z.hours.unique())==list(range(2,25,2))
        # Symbols at the same leads as in d,e (DO 2/10/18 h, PI 4/12/20 h, ...).
        for i,(ax,m) in enumerate(zip(cax,METHODS)):
            curves(ax,z[z.method==m],m,'hours','RMSE',5,markers=(i,4));facet_label(ax,STYLE[m][0].replace(' projected','\nprojected'))
            ax.set(xticks=LEAD24,xlim=(1.5,24.5))
            if i:ax.tick_params(labelleft=False)
        cax[0].set_ylabel('RMSE (TECU)');panel(cax[0],'c','Lead dependence')
        for j in range(2):label_strip(fig,gs[2,2*j:2*j+2],'Forecast lead (h)')
        fig.legend(handles=[handle(m) for m in METHODS],loc='outside upper center',ncol=4,columnspacing=1.2)
        # d,e: the same pooled lead errors as c, paired by seed; one shared scale symmetric about zero.
        lim=0
        for ax,(m,ref),letter,title in zip([dax,eax],[(METHODS[1],METHODS[0]),(METHODS[3],METHODS[2])],'de',['TEC-only: PI − DO','TEC + drivers: PI − DE']):
            lim=max(lim,delta_curves(ax,paired_lead_delta(z,'method',m,ref,'hours','RMSE',5),m,markers=(METHODS.index(m),4)))
            ax.set(xlabel='Forecast lead (h)',xticks=LEAD24,xlim=(1.5,24.5));panel(ax,letter,title)
        dax.set(ylim=(-1.1*lim,1.1*lim),ylabel='ΔRMSE (TECU)');dax.yaxis.set_major_locator(MaxNLocator(5,steps=[1,2,2.5,5,10]));eax.tick_params(labelleft=False)
        print('Figure 2 means',df.groupby('method').RMSE.mean().to_dict())
        return fig

    def figure_3(self):
        yr=self.read('china/year_metrics.csv');unique(yr,['model','seed','year'])
        cal=self.read('china/calibration_summary.csv');cal=cal[cal.activity=='All']
        fig=plt.figure(figsize=(MAIN_WIDTH,4.6),layout='constrained')
        # b-d: one period per column and one model per row (DO above adaptive PI), all six facets on one
        # coverage scale; a spans the grid so the outer edges align.
        gs=fig.add_gridspec(4,3,height_ratios=[1.15,.52,.52,.12])
        a=fig.add_subplot(gs[0,:]);methods=['DO','PI_mechanism_legacy','PI_static_legacy']
        for i,m in enumerate(methods):
            for j,year in enumerate([2015,2019,2024]):
                z=seeds(yr[(yr.model==m)&(yr.year==year)],10);points(a,j+(i-1)*.27,z.rmse_tecu,m,sd=True,spread=.08)
        a.set(xticks=range(3),xticklabels=['2015 (full year)','2019','2024'],ylabel='RMSE (TECU)',xlim=(-.5,2.5));panel(a,'a','Regional annual errors')
        a.legend(handles=[handle(m) for m in methods],loc='upper left',ncol=3,columnspacing=1.1)
        roles=['development_2015_h2','retrospective_2019','retrospective_2024']
        grid=[[],[]]
        for r in range(2):
            for j in range(3):grid[r].append(fig.add_subplot(gs[1+r,j],sharey=grid[0][0] if r or j else None,sharex=grid[0][j] if r else None))
        for j,(role,letter,title) in enumerate(zip(roles,'bcd',['2015 H2','2019','2024'])):
            for i,m in enumerate(methods[:2]):
                ax=grid[i][j];z=cal[(cal.model==m)&(cal.role==role)].sort_values('lead');assert len(z)==12
                name,key,marker,ls=STYLE[m];ax.plot(z.lead_hours,z.empirical_coverage_seed_mean*100,c=C[key],ls=ls,lw=MEAN_LW,marker=marker,ms=MS,
                                                     mec='white',mew=MEW,markevery=(i,3),label=name)
                ax.axhline(95,c='.5',ls=':',lw=REF_LW);ax.set(ylim=(88,100.7),xlim=(1,25),xticks=LEAD24,yticks=[88,92,96,100])
                if j:ax.tick_params(labelleft=False)
                if not i:ax.tick_params(labelbottom=False)
            panel(grid[0][j],letter,title)
        for ax,m in zip([grid[0][0],grid[1][0]],methods[:2]):ax.set_ylabel('Coverage (%)');facet_label(ax,STYLE[m][0])
        label_strip(fig,gs[3,:],'Forecast lead (h)')
        return fig

    def case(self):
        case=json.loads((self.source/'events/figure5_case_selected.json').read_text());ident=self.read('methods/regional_training_identity.csv')
        for slug in case['paper_slugs']:
            run_id=Path(case['run_dirs'][slug]).name.split('-')[-1]
            z=ident[(ident.run_id==run_id)&(ident.paper_slug==slug)&(ident.seed==42)];assert len(z)==1
        assert case['selected_case_id']=='doy173_174'
        return case

    def figure_4(self):
        case=self.case();df=self.read('events/figure5_timeseries_seed42.csv')
        fig=plt.figure(figsize=(MAIN_WIDTH,4.5),layout='constrained');gs=fig.add_gridspec(2,2,width_ratios=[1.6,1],wspace=.13)
        axes=[fig.add_subplot(gs[j,0]) for j in range(2)];event=fig.add_subplot(gs[:,1])
        models=['PI_mechanism_legacy','DO','DE_raw_concat']
        for ax,station,letter,title in zip(axes,['beijing','haikou'],'ab',['Beijing grid cell','Haikou grid cell']):
            z=df[df.station==station].sort_values('time_unix');assert len(z)==24;assert np.all(np.diff(z.time_unix)==7200)
            hours=(z.time_unix-z.time_unix.min())/3600
            ax.plot(hours,z.tec_truth,color=C['reference'],lw=.9,marker='.',ms=3.5,label='Processed reference')
            errors={}
            for slug,m in zip(case['paper_slugs'],models):
                name,key,marker,ls=STYLE[m];v=z['tec_'+slug];errors[name]=float(np.sqrt(np.mean((v-z.tec_truth)**2)))
                ax.plot(hours,v,color=C[key],ls=ls,lw=MEAN_LW,marker=marker,ms=MS,mec='white',mew=MEW,markevery=3,label=name)
            print('Figure 4',station,'cell RMSE',errors)
            for hour in [0,24]:ax.axvline(hour,c='.55',ls='--',lw=REF_LW,zorder=0)
            ax.axvline(28,c='.4',ls=':',lw=REF_LW,zorder=0)
            ax.set(xlim=(-1,47),xticks=[0,12,24,36,46],xticklabels=['22 Jun\n00','12','23 Jun\n00','12','22'],ylabel='TEC (TECU)');panel(ax,letter,title)
        axes[1].set_xlabel('Time in June 2015 (UTC)')
        axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.53),ncol=2,columnspacing=1,handlelength=2)
        ev=self.read('events/event_full_domain_summary.csv');unique(ev,['role','event_id','model'])
        p=ev.pivot(index=['role','event_id'],columns='model',values='rmse_tecu_seed_mean');assert len(p)==54 and p[['DO','PI_mechanism_legacy']].notna().all().all()
        role_names={'calibration_2015_h1':'2015 H1\ncalibration','evaluation_2015_h2':'2015 H2','development_evaluation_2015_h2':'2015 H2','robustness_2019':'2019','robustness_2024':'2024'}
        # c: paired event differences against their mean (Bland-Altman); on a 1:1 plot most differences are smaller than a symbol.
        d=p.PI_mechanism_legacy-p.DO;avg=(p.PI_mechanism_legacy+p.DO)/2;assert d.notna().all() and len(d)==54
        print('Figure 4c PI lower in',int((d<0).sum()),'of',len(d),'events; mean PI-DO by role',d.groupby(level=0).mean().round(4).to_dict())
        for i,(role,z) in enumerate(d.groupby(level=0,sort=True)):
            event.scatter(avg.loc[z.index],z,s=10,marker=['o','s','^','D'][i],facecolors='none',edgecolors=['#506879','#9C7749','#517362','#80738D'][i],lw=.5,label=role_names[role])
        lim=1.1*float(np.abs(d).max())
        event.axhline(0,c='.45',lw=REF_LW,zorder=0)
        event.set(xlim=(0,1.07*avg.max()),ylim=(-lim,lim),xlabel='Mean of DO and PI\nRMSE (TECU)',ylabel='PI − DO RMSE (TECU)',xticks=[0,1,2,3])
        event.yaxis.set_major_locator(MaxNLocator(5,steps=[1,2,2.5,5,10]));event.set_box_aspect(1);panel(event,'c','54 regional events')
        event.legend(loc='upper center',bbox_to_anchor=(.5,-.33),handletextpad=.5,labelspacing=.55)
        return fig

    def figure_5(self):
        df=self.read('fusion/per_run_role_metrics.csv');summary=self.read('fusion/five_seed_summary.csv')
        fig,axes=plt.subplots(1,2,figsize=(MAIN_WIDTH,3.2),layout='constrained',sharey=True)
        for ax,dataset,role,base,letter,title in zip(axes,['gim','china'],['evaluation_2015_2019_2020_2021_2022','retrospective_2024'],['gim_de_film_c06','china_de_film_c07'],'ab',['Global 2015,\n2019–2022','Regional 2024']):
            for i,(suffix,label) in enumerate(VARIANTS):
                z=seeds(df[(df.dataset==dataset)&(df.role==role)&(df.candidate_id==base+suffix)],5);v=z.RMSE.to_numpy()
                s=summary[(summary.dataset==dataset)&(summary.role==role)&(summary.candidate_id==base+suffix)];assert len(s)==1
                np.testing.assert_allclose([v.mean(),v.std(ddof=1)],[s.iloc[0].RMSE_mean,s.iloc[0].RMSE_sample_sd],rtol=1e-12)
                ax.scatter(v,i+np.linspace(-.14,.14,5),marker=['o','s','^','D'][i],s=15,c='#526775',zorder=3)
                ax.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt='|',ms=12,mew=MEAN_LW,color=C['reference'],elinewidth=.6,capthick=.6,capsize=3,zorder=4)
            ax.set(yticks=range(4),yticklabels=[x[1] for x in VARIANTS],xlabel='RMSE (TECU)',ylim=(3.5,-.6));panel(ax,letter,title);ax.xaxis.set_major_locator(MaxNLocator(4))
        return fig

    def figure_6(self):
        df=self.read('h96/per_seed_year_group_metrics.csv')
        # Height leaves the float page its original margin with the one-line-longer legend.
        fig=plt.figure(figsize=(MAIN_WIDTH,4.96),layout='constrained')
        # a,b: one model per facet; facets share the vertical scale within a period and the lead axis within a model.
        gs=fig.add_gridspec(4,6,height_ratios=[1,1,.12,1]);rows={};grid=[]
        for r in range(2):
            grid.append([])
            for k in range(3):grid[r].append(fig.add_subplot(gs[r,2*k:2*k+2],sharey=grid[r][0] if k else None,sharex=grid[0][k] if r else None))
        for axes,group,letter,title in zip(grid,['old','extension'],'ab',['2015 and 2019–2022','2023–2025 extension']):
            for k,(ax,m,method) in enumerate(zip(axes,METHODS[:3],['DO-TEC-only-H96','PI-TEC-only-H96','DE-ZScore-H96'])):
                z=df[(df.group==group)&(df.candidate_id==method)&df.selection.str.fullmatch(r'lead_\d+h')].copy()
                z['hours']=z.selection.str.extract(r'lead_(\d+)h').astype(int);assert sorted(z.hours.unique())==list(range(2,97,2))
                np.testing.assert_allclose(z.RMSE,np.sqrt(z.SSE/z.n),rtol=1e-12)
                curves(ax,z,m,'hours','RMSE',3,markers=[11,23,35,47]);rows[group,method]=z
                ax.set(xlim=(0,98),xticks=LEAD96)
                for x in [24,48,72,96]:ax.axvline(x,color='.75',ls=':',lw=REF_LW,zorder=0)
                if k:ax.tick_params(labelleft=False)
                if group=='old':ax.tick_params(labelbottom=False)
            axes[0].set_ylabel('RMSE (TECU)');panel(axes[0],letter,title)
        label_strip(fig,gs[2,:],'Forecast lead (h)')
        fig.legend(handles=[handle(m) for m in METHODS[:3]],loc='outside upper center',ncol=3,columnspacing=1)
        # c,d: PI minus DO paired by seed within each period; the two scales differ, each symmetric about zero.
        cd=[]
        for j,(group,letter,title) in enumerate(zip(['old','extension'],'cd',['PI − DO, 2015 and 2019–2022','PI − DO, 2023–2025'])):
            ax=fig.add_subplot(gs[3,3*j:3*j+3]);cd.append(ax);z=pd.concat([rows[group,'PI-TEC-only-H96'],rows[group,'DO-TEC-only-H96']])
            lim=1.1*delta_curves(ax,paired_lead_delta(z,'candidate_id','PI-TEC-only-H96','DO-TEC-only-H96','hours','RMSE',3),METHODS[1],markers=[11,23,35,47])
            ax.set(xlim=(0,98),xticks=LEAD96,ylim=(-lim,lim),xlabel='Forecast lead (h)',ylabel='ΔRMSE (TECU)');panel(ax,letter,title)
            ax.yaxis.set_major_locator(MaxNLocator(5,steps=[1,2,2.5,5,10]))
            for x in [24,48,72,96]:ax.axvline(x,color='.75',ls=':',lw=REF_LW,zorder=0)
        # d's axis labels fall inside the middle facet columns, which would widen that facet. After the
        # constrained layout, give the three facets equal widths between the outer edges of c and d, then freeze.
        fig.align_ylabels([grid[0][0],grid[1][0],cd[0]]);fig.get_layout_engine().execute(fig)
        left,right=cd[0].get_position().x0,cd[1].get_position().x1;gap=10/72/fig.get_figwidth();w=(right-left-2*gap)/3
        for axes in grid:
            for k,ax in enumerate(axes):p=ax.get_position();ax.set_position([left+k*(w+gap),p.y0,w,p.height])
        fig.set_layout_engine('none')
        # The legend becomes column headers: each model's entry centred over its facets, in the band the
        # layout reserved for it (c,d reuse the PI style). In-axes labels would cross the dotted lead guides.
        legend=fig.legends[0];top=legend.get_window_extent(fig.canvas.get_renderer()).y1/fig.bbox.height;legend.remove()
        for k,m in enumerate(METHODS[:3]):
            p=grid[0][k].get_position();fig.legend(handles=[handle(m)],loc='upper center',bbox_to_anchor=(p.x0+p.width/2,top),borderaxespad=0)
        return fig

    def figure_S1(self):
        lead=self.read('global_historical/lead_metrics.csv');lat=self.read('global_historical/latitude_band_metrics.csv')
        fig=plt.figure(figsize=(SI_WIDTH,5.4),layout='constrained')
        # a: one model per facet on a shared scale, each headed by its legend entry and seed count.
        gs=fig.add_gridspec(3,4,height_ratios=[1,.1,1]);fa=[fig.add_subplot(gs[0,0])];fa+=[fig.add_subplot(gs[0,k],sharey=fa[0]) for k in range(1,4)]
        b=fig.add_subplot(gs[2,:])
        bands=['abs_lat_0_5','abs_lat_5_20','abs_lat_20_40','abs_lat_40_60','abs_lat_60_90']
        for i,(ax,m) in enumerate(zip(fa,HISTORICAL)):
            n=10 if i<2 else 3;curves(ax,lead[lead.model==m],m,'lead_hours','rmse_tecu',n,markers=(i,4))
            ax.set(xticks=LEAD24,xlim=(1.5,24.5));column_header(ax,handle(m,label=STYLE[m][0]+f' (n={n})'))
            if i:ax.tick_params(labelleft=False)
            for j,band in enumerate(bands):
                z=seeds(lat[(lat.model==m)&(lat.geographic_latitude_band==band)],n)
                points(b,j+(i-1.5)*.22,z.rmse_tecu,m,sd=True,spread=.07 if n==10 else .03,sd_gap=.04)
        fa[0].set_ylabel('RMSE (TECU)');panel(fa[0],'a','Original-year lead errors');label_strip(fig,gs[1,:],'Forecast lead (h)')
        b.set(xticks=range(5),xticklabels=['0–5','5–20','20–40','40–60','60–90'],xlabel='Absolute geographic latitude (°)',ylabel='RMSE (TECU)');panel(b,'b','Latitude-band errors')
        return fig

    def figure_S2(self):
        monthly=self.read('global_historical/figureS1_monthly_rmse.csv');amp=self.read('global_historical/figureS2_amplitude_bins.csv')
        slugs=['do_gwn','a1_pinn_confirm_pi_pinn_drvonly_adv_lp0p10','st_gpt_stable','iri_2020']
        # a: one model per facet on a shared scale (the deterministic IRI-2020 reference has its own facet);
        # b,c span two facet columns each, so their edges align with the facets.
        fig=plt.figure(figsize=(SI_WIDTH,5.2),layout='constrained');gs=fig.add_gridspec(3,4,height_ratios=[1,.1,1])
        fa=[fig.add_subplot(gs[0,0])];fa+=[fig.add_subplot(gs[0,k],sharey=fa[0]) for k in range(1,4)];axes=[fig.add_subplot(gs[2,2*j:2*j+2]) for j in range(2)]
        for ax,slug,m in zip(fa,slugs[:3],['DO','PI_mechanism_legacy','ST_GPT_stable']):
            z=monthly[monthly.paper_slug==slug];assert set(z.seed)=={42,1024,2025};assert set(z.month)==set(range(1,13));curves(ax,z,m,'month','rmse',3,markers=list(range(12)))
            column_header(ax,handle(m))
        iri=monthly[monthly.paper_slug==slugs[-1]];unique(iri,['seed','month']);assert (iri.groupby('month').rmse.nunique()==1).all()
        z=iri.drop_duplicates('month').sort_values('month');style=dict(c=C['reference'],ls='-.',marker='x',ms=3,mew=MEAN_LW,lw=MEAN_LW)
        fa[3].plot(z.month,z.rmse,**style);column_header(fa[3],Line2D([],[],**style,label='IRI-2020\n(deterministic)'))
        for k,ax in enumerate(fa):
            ax.set(xticks=[3,6,9,12],xlim=(.5,12.5))
            if k:ax.tick_params(labelleft=False)
        fa[0].set_ylabel('RMSE (TECU)');panel(fa[0],'a','Original-year monthly errors');label_strip(fig,gs[1,:],'Calendar month')
        counts=amp.groupby(['bin_lo','bin_hi']).n.agg(['nunique','first']);assert (counts['nunique']==1).all()
        print('Figure S2 bins and counts',counts.to_string())
        bins=sorted(amp.bin_lo.unique());assert bins==[0,20,40,60];assert np.isinf(amp.loc[amp.bin_lo==60,'bin_hi']).all()
        labels=['0–20','20–40','40–60','60+']
        for ax,metric,letter in zip(axes,['rmse','mae'],'bc'):
            for slug,m in zip(slugs[:2],['DO','PI_mechanism_legacy']):
                z=amp[amp.paper_slug==slug].copy();z['bin']=z.bin_lo.map({x:i for i,x in enumerate(bins)});assert set(z.seed)=={42,1024,2025}
                # Bin categories have no continuous x distance: show individual
                # points and mean bars, not a fictitious interpolated TEC curve.
                for i in range(4):
                    v=seeds(z[z.bin==i],3)[metric];points(ax,i+(-.13 if m=='DO' else .13),v,m,spread=.055,mean_half=.09)
            ax.set(xticks=range(4),xticklabels=labels,xlabel='Reference TEC bin (TECU)',ylabel=metric.upper()+' (TECU)',xlim=(-.5,3.5));panel(ax,letter,'Amplitude '+metric.upper())
        return fig

    def figure_S3(self):
        case=self.case();z=np.load(self.source/'events/figure5_snapshots_seed42.npz')
        assert str(z['time_utc'])=='2015-06-23T04:00:00Z'
        lat,lon=z['lat_grid'],z['lon_grid'];assert lat.shape==lon.shape==(41,71)
        np.testing.assert_allclose(lat[:,0],np.arange(15,56));np.testing.assert_allclose(lon[0,:],np.arange(70,141))
        fields=[z['truth']]+[z[k]-z['truth'] for k in case['paper_slugs']]
        vmax=max(float(np.max(np.abs(f))) for f in fields[1:]);print('S3 full signed error range',-vmax,vmax,'reference range',float(fields[0].min()),float(fields[0].max()))
        fig=plt.figure(figsize=(SI_WIDTH,4.6),layout='constrained');gs=fig.add_gridspec(2,4,width_ratios=[1,.045,1,.045],wspace=.12)
        axes=[fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,2),(1,0),(1,2)]];cbref=fig.add_subplot(gs[0,1]);cberr=fig.add_subplot(gs[:,3])
        for i,(ax,field,title,letter) in enumerate(zip(axes,fields,['Processed reference','PI − reference','DO − reference','Raw DE − reference'],'abcd')):
            im=ax.pcolormesh(lon,lat,field,shading='nearest',cmap='viridis' if i==0 else 'RdBu_r',vmin=0 if i==0 else -vmax,vmax=field.max() if i==0 else vmax,rasterized=True)
            # Native one-degree cells, no interpolation or geographic resampling.
            ax.scatter([116,110],[40,20],s=22,marker='x',color='white',lw=2)
            ax.scatter([116,110],[40,20],s=20,marker='x',color=C['reference'],lw=.8)
            ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)',xlim=(70,140),ylim=(15,55),xticks=[70,90,110,130],yticks=[15,25,35,45,55]);panel(ax,letter,title)
            if i==0:fig.colorbar(im,cax=cbref,label='Reference TEC (TECU)')
            if i==3:fig.colorbar(im,cax=cberr,label='Prediction − reference (TECU)')
        return fig

    def figure_S4(self):
        df=self.read('diagnostics/driver/driver_consistency_runs.csv');case=self.case();slugs=case['paper_slugs'];lookup=dict(zip(['PI_mechanism_legacy','DO','DE_raw_concat'],slugs))
        metrics=['score_corr_roti_dstn','score_corr_absrot_dstn','score_corr_roti_kp','score_corr_absrot_kp','score_auc_roti','score_auc_absrot']
        titles=['TSI versus −Dst','|ROT| versus −Dst','TSI versus Kp','|ROT| versus Kp','TSI storm/quiet AUC','|ROT| storm/quiet AUC']
        fig,axes=plt.subplots(2,3,figsize=(SI_WIDTH,4.6),layout='constrained')
        for ax,metric,title,letter in zip(axes.flat,metrics,titles,'abcdef'):
            for i,m in enumerate(['DO','PI_mechanism_legacy','DE_raw_concat']):
                z=seeds(df[df.paper_slug==lookup[m]],10);points(ax,i,z[metric],m,sd=True,spread=.15,sd_gap=.08)
            ax.axhline(0,c='.7',lw=REF_LW,zorder=0);ax.set(xticks=[0,1,2],xticklabels=['DO','PI\nadaptive','Raw DE'],ylabel='Agreement score',xlim=(-.5,2.5));panel(ax,letter,title)
        return fig

    def figure_S5(self):
        df=self.read('methods/regional_weight_trajectories.csv').drop_duplicates();identity=self.read('methods/regional_training_identity.csv')
        unique(df,['run_id','term','log_step'])
        # Some distinct recorded steps share an integer epoch. Preserve both
        # values at that x coordinate (vertical segment); never average them.
        assert df.run_id.nunique()==19
        # Training metadata covers 18/19 recovered trajectories. The remaining
        # adaptive run is independently bound by the retained driver run table.
        bindings=self.read('diagnostics/driver/driver_consistency_runs.csv')
        bindings['run_id']=bindings.run_dir.map(lambda p: Path(p).name.split('-')[-1])
        adaptive='regfam_flagship__prod5__mech_agg_mrmse_plateau10_es30';static='regfam_flagship__prod5__pi_static_combo'
        assert set(df.paper_slug)=={adaptive,static}
        # a-c: terms active only in the adaptive configuration. d-g: terms shared with the static configuration,
        # the ten adaptive runs above the nine static runs on the same per-term scale, so the sets no longer overlap.
        terms=['adv','chem','night','diff','nonneg','rot','roti'];names=['Gradient alignment','Production–loss','Night growth','Graph temporal','Nonnegativity','ROT','TSI']
        fig=plt.figure(figsize=(SI_WIDTH,6.3),layout='constrained');gs=fig.add_gridspec(5,4,height_ratios=[1,.12,1,1,.12])
        top=[fig.add_subplot(gs[0,k]) for k in range(3)];mid=[fig.add_subplot(gs[2,k]) for k in range(4)]
        low=[fig.add_subplot(gs[3,k],sharex=mid[k],sharey=mid[k]) for k in range(4)]
        for k,(term,title,letter) in enumerate(zip(terms,names,'abcdefg')):
            z=df[df.term==term];shared=k>=3;n={False:0,True:0}
            for run,g in z.groupby('run_id',sort=True):
                g=g.sort_values(['raw_recorded_epoch','log_step']);assert len(g.paper_slug.unique())==1
                ident=identity[identity.run_id==run]
                if ident.empty:ident=bindings[bindings.run_id==run]
                assert len(ident)==1 and g.seed.iloc[0]==ident.seed.iloc[0] and g.paper_slug.iloc[0]==ident.paper_slug.iloc[0]
                isstatic=g.paper_slug.iloc[0]==static;n[isstatic]+=1
                # Opaque thin runs, one configuration per facet.
                ax=(low if isstatic else mid)[k-3] if shared else top[k]
                ax.plot(g.raw_recorded_epoch,g.log_variance,color=C['static' if isstatic else 'pi'],ls='--' if isstatic else '-',lw=SEED_LW)
            assert n=={False:10,True:9 if shared else 0},(term,n)
            panel(mid[k-3] if shared else top[k],letter,title)
        for ax in top+mid+low:ax.xaxis.set_major_locator(MaxNLocator(3))
        for ax in mid:ax.tick_params(labelbottom=False)
        for ax,text in zip([top[0],mid[0],low[0]],['Adaptive (n=10)','Adaptive (n=10)','Static (n=9)']):ax.set_ylabel('Log variance');facet_label(ax,text,top=True)
        for r in [1,4]:label_strip(fig,gs[r,:],'Recorded epoch')
        key=fig.add_subplot(gs[0,3]);key.axis('off')
        key.legend(handles=[Line2D([],[],color=C['pi'],ls='-',lw=MEAN_LW,label='PI adaptive'),Line2D([],[],color=C['static'],ls='--',lw=MEAN_LW,label='PI static')],loc='center',ncol=1)
        print('S5 actual trajectories',df.groupby('term').run_id.nunique().to_dict(),'minimum recorded epoch',int(df.raw_recorded_epoch.min()))
        return fig


def build_figures(source,out,only=None,legacy_root=None):
    source,out=Path(source),Path(out);out.mkdir(parents=True,exist_ok=True)
    # Only import genuinely missing approved small inputs. Existing canonical
    # files (including sanitized provenance fields) always take precedence.
    if legacy_root:
        for dest,rel in LEGACY_FILES.items():
            if not (source/dest).exists():
                original=Path(legacy_root)/rel
                if original.exists():
                    (source/dest).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,source/dest)
    font=setup();print('Physical widths (mm)',MAIN_WIDTH*25.4,SI_WIDTH*25.4,'font',font)
    ids=['2','3','4','5','6','S1','S2','S3','S4','S5']
    selected=ids if only is None else [s.strip() for s in only.split(',')]
    if not set(selected)<=set(ids):raise ValueError('Allowed numerical figures: '+','.join(ids))
    dest=out/'figures';preview=out/'preview';dest.mkdir(exist_ok=True);preview.mkdir(exist_ok=True)
    builder=Figures(source)
    for number in selected:
        fig=getattr(builder,'figure_'+number)();fig.canvas.draw()
        # A deterministic basic check; the exported pages are also inspected
        # visually at final manuscript width before promotion to figures/.
        renderer=fig.canvas.get_renderer();bounds=fig.bbox
        for ax in fig.axes:
            for item in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.texts]:
                if item.get_visible() and item.get_text():
                    box=item.get_window_extent(renderer)
                    if box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.width+1 or box.y1 > bounds.height+1:
                        raise ValueError(f'Figure {number}: clipped text {item.get_text()}')
        # Journal floors at the printed size: text at least 8 pt; visible strokes at least 0.3 pt.
        for item in fig.findobj(Text):
            if item.get_visible() and item.get_text().strip() and item.get_fontsize()<8:
                raise ValueError(f'Figure {number}: {item.get_fontsize():g} pt text {item.get_text()!r}')
        for item in fig.findobj(lambda o:isinstance(o,(Line2D,Collection)) and o.get_visible()):
            if isinstance(item,Line2D):
                widths=[item.get_linewidth()] if item.get_linestyle() not in ('None','',' ') else []
                if item.get_marker() not in (None,'None','',' '):widths.append(item.get_markeredgewidth())
            else:
                # Empty collections (e.g. an undrawn colorbar divider) paint nothing.
                edges=to_rgba_array(item.get_edgecolor())
                widths=list(np.atleast_1d(item.get_linewidth())) if len(edges) and edges[:,3].max()>0 and len(item.get_paths()) else []
            if any(0<w<.3 for w in widths):raise ValueError(f'Figure {number}: {min(widths):g} pt stroke in {item}')
        name='Figure_'+number
        for ext in ['pdf','svg']:fig.savefig(dest/(name+'.'+ext),dpi=300)
        fig.savefig(preview/(name+'.png'),dpi=300)
        with Image.open(preview/(name+'.png')) as im:ImageOps.grayscale(im).save(preview/(name+'_gray.png'))
        plt.close(fig);print('Saved',name,flush=True)
    return {'numerical_figures':selected,'font':font,'figure1':'PowerPoint schematic supplied separately'}
