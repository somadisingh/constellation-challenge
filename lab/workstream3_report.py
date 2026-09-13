"""Summarize frozen branch-fusion experiments without choosing on confirmation."""
import json
from pathlib import Path
import numpy as np
from lab.branch_agreement import POLICIES
from lab.imagebench.generate import write_json,sha


def summarize(root):
    root=Path(root);out={}
    for name in ['train','v1_development','v2_development','v2_calibration']:
        path=root/name/'report.json'
        if not path.exists():continue
        report=json.loads(path.read_text())
        baseline=report['metrics']['baseline'];n=len(baseline['scenes'])
        summaries={}
        for policy,ev in report['metrics'].items():
            gains=[ev['scenes'][s]['score']-baseline['scenes'][s]['score'] for s in baseline['scenes']]
            summaries[policy]=dict(mean=ev['mean'],delta={k:ev['mean'][k]-baseline['mean'][k] for k in ev['mean']},
                improved=sum(x>1e-10 for x in gains),regressed=sum(x< -1e-10 for x in gains),worst=ev['worst_score'],
                name_only=report['identification_only'][policy]['mean'])
        availability=[]
        for a in report['agreements']:
            scene=json.loads((root/name/(a['scene']+'.json')).read_text())
            ranks=[]
            for b in scene['branches']:
                names=[h['name'] for h in b['diag'].get('hypotheses',[]) if h.get('support',0)>=4 and 'nodes' in h]
                ranks.append(names.index(a['true'])+1 if a['true'] in names else None)
            availability.append(dict(scene=a['scene'],true_ranks=ranks))
        agreed=[a for a in report['agreements'] if a['agree']]
        out[name]=dict(n=n,true_class_rank_diagnostic=availability,report_sha256=sha(path),policies=summaries,agreement_n=len(agreed),
                     agreement_correct=sum(a['branches'][0]==a['true'] for a in agreed))
        if name=='train':
            folds=[]
            for held in baseline['scenes']:
                dev=[s for s in baseline['scenes'] if s!=held]
                # Fixed policy order resolves ties in favour of the incumbent.
                pick=max(POLICIES,key=lambda p:np.mean([report['metrics'][p]['scenes'][s]['score'] for s in dev]))
                folds.append(dict(held=held,policy=pick,metrics=report['metrics'][pick]['scenes'][held]))
            out[name]['loso']=dict(folds=folds,mean={k:float(np.mean([f['metrics'][k] for f in folds])) for k in baseline['mean']})
    write_json(root/'summary.json',out)
    for name,data in out.items():
        print(name,'n=',data['n'],'agreement',data['agreement_correct'],'/',data['agreement_n'])
        for policy,v in data['policies'].items():
            print(policy,round(v['mean']['score'],6),round(v['delta']['score'],6),v['mean']['identification'])
    return out

if __name__=='__main__':summarize('outputs/lab/workstream3')
