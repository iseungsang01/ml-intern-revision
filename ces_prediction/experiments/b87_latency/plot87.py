"""Plot B87 cached readout; does not access or fit shot data."""
import json
import common87 as M
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    source=M.C.PAPER/'b87_latency.json'
    d=json.loads(source.read_text(encoding='utf-8'))
    fig,axs=plt.subplots(1,5,figsize=(16,4),sharey=True)
    for ax,(name,rec) in zip(axs,d['designs'].items()):
        for prefix,label,color in [('H','History only','#555555'),('X','History + present fast','#1469a6')]:
            cells=[rec['metrics'][f'{prefix}{j}']['RMSE'] for j in (0,1,2,5)]
            point=np.array([c['point'] for c in cells])
            bounds=np.array([c['cluster'] for c in cells])
            ax.plot([0,10,20,50],point,'o-',color=color,label=label)
            ax.fill_between([0,10,20,50],bounds[:,0],bounds[:,1],color=color,alpha=.16)
        ax.set_title(name)
        ax.set_xlabel('CES timestamp horizon (ms)')
        ax.grid(alpha=.2)
    axs[0].set_ylabel('RMSE (eV)')
    axs[0].legend(fontsize=8)
    fig.suptitle('Matched regressions: present fast diagnostics and later CES timestamps')
    fig.text(.5,.015,'Source: b87_latency.json. Shading: 95% shot-cluster interval. Exploratory used-data analysis; exposure and processing add physical latency.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.92])
    for ext in ('png','pdf'):
        fig.savefig(M.C.PAPER/f'b87_latency.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    main()
