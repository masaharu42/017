"""Create an unapproved CNN feedback artifact for review; never train or approve."""
import argparse,json
from pathlib import Path
import torch
from scripts import train_mechanics_cnn as base
from physics.cnn_mechanics import SCHEMA,TARGETS,SingleScaleCNN,sha256

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--original-config',required=True);p.add_argument('--cnn-dir',required=True);p.add_argument('--output',required=True)
 a=p.parse_args();root,original=base.load(a.original_config); c=Path(a.cnn_dir)
 plan=json.loads((c/'plan.json').read_text());summary=json.loads((c/'summary.json').read_text())
 if plan['targets']!=TARGETS or plan['source_plan_sha256']!=sha256(root/'plan.json'):
  raise ValueError('CNN target or source plan mismatch')
 if sha256(c/'best.pt')!=summary['best_weights_sha256']:raise ValueError('CNN checkpoint hash mismatch')
 train=plan['train_ids']; val=plan['validation_ids']; all_ids=original['alloy_ids']
 if len(train)!=18 or len(val)!=3 or set(train)&set(val) or set(train+val)!=set(all_ids) or 3 in all_ids:
  raise ValueError('Unexpected CNN training/validation split')
 labels=Path(plan['labels']);
 if sha256(labels)!=plan['labels_sha256']:raise ValueError('CNN labels changed')
 checkpoint=torch.load(c/'best.pt',map_location='cpu',weights_only=True)
 model=SingleScaleCNN();model.load_state_dict(checkpoint['model'])
 if checkpoint['mean']!=plan['target_mean'] or checkpoint['scale']!=plan['target_scale']:
  raise ValueError('Checkpoint normalization differs from plan')
 for i in all_ids:
  path=Path(original['records'][str(i)]['image']);
  if sha256(path)!=original['hashes'].get(str(path)):raise ValueError(f'Image mismatch: {i}')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 weights=out/'cnn_weights.pt';torch.save(model.state_dict(),weights)
 artifact={'schema':SCHEMA,'target_names':TARGETS,'patch_size':256,'pixel_transform':'native_rgb_minus_one_one',
  'members':[{'weights_file':weights.name,'weights_sha256':sha256(weights),'target_mean':checkpoint['mean'],'target_scale':checkpoint['scale']}],
  'target_scale':checkpoint['scale'],'approved_targets':[False,False], 'training_alloy_ids':train,
  'labels_sha256':sha256(labels),'input_sha256':{original['records'][str(i)]['image']:original['hashes'][original['records'][str(i)]['image']] for i in all_ids},
  'source_cnn_plan_sha256':sha256(c/'plan.json'),'source_cnn_summary_sha256':sha256(c/'summary.json'),
  'source_best_sha256':sha256(c/'best.pt'),'feedback_alloy_ids_requested':all_ids,
  'cnn_validation_alloy_ids':val,'status':'pending_review_not_approved'}
 base.dump(out/'cnn_proxy.json',artifact)
 draft={'schema':'image_cnn_feedback_decision_v1','status':'pending_review','proxy_sha256':sha256(out/'cnn_proxy.json'),
  'feedback_alloy_ids':all_ids,'excluded_alloy_id':3,'requested_active_targets':[True,True],
  'requested_max_t':150,'group_size':4,'notes':'Draft only; prediction/domain/gradient evidence and signed review are missing. No authorization.'}
 base.dump(out/'decision_draft.json',draft)
 print('PENDING REVIEW only; no training authorized:',out.resolve())
if __name__=='__main__':main()
