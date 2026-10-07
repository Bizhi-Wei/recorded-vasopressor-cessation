from pathlib import Path
import csv, json, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((OUT/'figure1_source.csv').open()))
plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(7.2,3.5));fig.subplots_adjust(left=.09,right=.98,bottom=.22,top=.76,wspace=.35)
colors=['#0072B2','#D55E00']
for j,(drug,label,unit) in enumerate([('ne','Norepinephrine','Dose (mcg/kg/min)'),('va','Vasopressin','Dose (U/h)')]):
    ax=axs[j]
    for g,color in enumerate(colors):
        rr=[r for r in rows if r['drug']==drug and int(r['A'])==g]
        for r in rr:
            a,z=float(r['window_start']),float(r['window_end']);m,l,u=[float(r[k]) for k in ['median','q25','q75']]
            ax.fill_between([a,z],[l,l],[u,u],color=color,alpha=.16,lw=0)
            ax.plot([a,z],[m,m],color=color,lw=1.8,ls='-' if g==0 else '--')
            ax.scatter([(a+z)/2],[m],color=color,s=22,marker='o' if g==0 else 's',zorder=3)
    ax.set(xlim=(-6,0),xticks=[-6,-4,-2,0],xlabel='Hours before recorded cessation',ylabel=unit)
    ax.set_ylim(0,.13 if j==0 else 2.75)
    ax.set_title(label,loc='left',fontsize=11,pad=10)
    ax.text(-.16,1.10,'ab'[j],transform=ax.transAxes,fontweight='bold',fontsize=12)
    ax.grid(axis='y',color='#eeeeee',lw=.5);ax.set_axisbelow(True)
fig.legend([Patch(facecolor=c,alpha=.6) for c in colors],['Norepinephrine-first (n = 192)','Vasopressin-first (n = 92)'],loc='upper center',bbox_to_anchor=(.52,.99),ncol=2,frameon=False,fontsize=9)
fig.text(.5,.03,'Two-hour time-weighted doses: median and interquartile range across admissions',ha='center',fontsize=8)
fig.canvas.draw()
for ext,kw in [('svg',{}),('pdf',{}),('tiff',{'dpi':600,'pil_kwargs':{'compression':'tiff_lzw'}}),('png',{'dpi':180})]:fig.savefig(OUT/('Figure1.'+ext),**kw)
# Check that all source panels use complete frozen groups and vector text.
assert all(int(r['n_full'])==int(r['n_group']) for r in rows)
svg=(OUT/'Figure1.svg').read_text();assert '<text' in svg and '<image' not in svg
qa={'source_rows':len(rows),'groups':[192,92],'width_inches':7.2,'height_inches':3.5,'tiff_dpi':600,'svg_live_text':True,'svg_embedded_raster':False,'visual_review':'PENDING','source_sha256':hashlib.sha256((OUT/'figure1_source.csv').read_bytes()).hexdigest()}
(OUT/'figure_validation.json').write_text(json.dumps(qa,indent=2));print(qa)
