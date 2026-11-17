import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import yaml

from physics.cnn_mechanics import SingleScaleCNN, CNNMechanics, SCHEMA, TARGETS, sha256
from scripts import train_mechanics_cnn as training
from ebsd_diffusion.mechanics_runtime import feedback_gradient_modules, target_losses


def proxy(tmp_path):
    torch.manual_seed(12)
    model = SingleScaleCNN(); torch.save(model.state_dict(), tmp_path/'model.pt')
    artifact = dict(schema=SCHEMA,target_names=TARGETS,patch_size=16,
        pixel_transform='native_rgb_minus_one_one', approved_targets=[True,True],target_scale=[20.,2.],
        members=[dict(weights_file='model.pt',weights_sha256=sha256(tmp_path/'model.pt'),
                      target_mean=[100.,10.],target_scale=[20.,2.])])
    (tmp_path/'cnn_proxy.json').write_text(json.dumps(artifact))
    return CNNMechanics.load(tmp_path/'cnn_proxy.json')


def test_frozen_cnn_decoder_backprop_updates_only_generator(tmp_path):
    cnn=proxy(tmp_path);cnn.train()  # exported teacher stays eval
    assert not cnn.training and not any(m.training for m in cnn.modules())
    before={k:v.clone() for k,v in cnn.state_dict().items()}
    decoder=torch.nn.Conv2d(3,3,1).requires_grad_(False).eval()
    decoder_before=deepcopy(decoder.state_dict())
    generator=torch.nn.Conv2d(3,3,1);condition=torch.nn.Linear(1,3)
    optimizer=torch.optim.AdamW(list(generator.parameters())+list(condition.parameters()),lr=.01)
    noise=torch.randn(4,3,16,16)
    image=decoder(generator(noise)+condition(torch.ones(4,1))[:,:,None,None])
    values=cnn(image).reshape(2,2,2).mean(1)
    loss,_=target_losses(values,torch.tensor([[110.,12.],[120.,13.]]),torch.ones(2,2,dtype=torch.bool),
                          cnn.target_scale,cnn.active,torch.ones(2))
    loss.backward()
    assert generator.weight.grad.abs().sum()>0 and condition.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in cnn.parameters())
    assert decoder.weight.grad is None
    old=generator.weight.clone();optimizer.step()
    assert not torch.equal(old,generator.weight)
    assert all(torch.equal(v,cnn.state_dict()[k]) for k,v in before.items())
    assert all(torch.equal(v,decoder.state_dict()[k]) for k,v in decoder_before.items())
    assert torch.allclose(values.detach(),cnn(image.detach()).reshape(2,2,2).mean(1))


def test_cnn_artifact_rejects_modified_weights(tmp_path):
    proxy(tmp_path)
    (tmp_path/'model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):CNNMechanics.load(tmp_path/'cnn_proxy.json')


def test_runtime_cnn_path_uses_images_without_compositions(tmp_path):
    from types import SimpleNamespace
    from ebsd_diffusion.mechanics_runtime import MechanicsRuntime
    runtime=object.__new__(MechanicsRuntime)
    runtime.cfg={'group_size':2,'max_decoded_windows':4}
    runtime.is_cnn=True;runtime.proxy=proxy(tmp_path);runtime.active=[True,True]
    runtime.max_t=3;runtime.calibrate=True
    conv=torch.nn.Conv2d(3,3,1)
    decoder=torch.nn.Conv2d(3,3,1).requires_grad_(False)
    vae=SimpleNamespace(decode=lambda z:SimpleNamespace(sample=decoder(z)))
    noisy=torch.randn(4,3,16,16);pred=conv(noisy)
    batch={'mechanics_valid':torch.ones(4,2,dtype=torch.bool),
           'mechanics_target':torch.tensor([[110.,12.]]).repeat(4,1)}
    loss,_,info=runtime.loss(vae,noisy,pred,torch.ones(4,dtype=torch.long),
        torch.zeros(4,dtype=torch.bool),batch,torch.tensor([.99,.9,.8,.7]),1.)
    loss.backward()
    assert info['groups']==2 and info['image_gradient_norm']>0
    assert conv.weight.grad.abs().sum()>0 and decoder.weight.grad is None
    assert info['boundary_saturated_fraction'] is None


def test_gradient_inventory_covers_each_parameter_once():
    from diffusers import UNet2DModel
    unet=UNet2DModel(sample_size=8,in_channels=4,out_channels=4,block_out_channels=(32,32),
        down_block_types=('DownBlock2D','DownBlock2D'),up_block_types=('UpBlock2D','UpBlock2D'),layers_per_block=1)
    modules=feedback_gradient_modules(unet,torch.nn.Linear(1,128))
    assert 'unet.down_blocks.0' in modules and 'unet.mid_block' in modules and 'unet.up_blocks.1' in modules


@pytest.fixture
def fixture(tmp_path):
    ids=[1,2,4,5,6,21]; rows=[]; manifest=[]; hashes={}
    rng=np.random.default_rng(3)
    for i in ids:
        p=tmp_path/f'{i}.png';Image.fromarray(rng.integers(0,255,(24,24,3),dtype=np.uint8)).save(p)
        hashes[str(p)]=sha256(p);manifest.append(dict(alloy_id=i,image_path=str(p)))
        rows.append(dict(alloy_id=i,peak_stress_MPa=100.+i,endpoint_strain_pct=10.+i*.2))
    manifest.append(dict(alloy_id=3,image_path='/DO_NOT_READ.png'))
    extraction=tmp_path/'extraction';extraction.mkdir()
    pd.DataFrame(rows).to_csv(extraction/'alloy_features.csv',index=False)
    meta=dict(training_alloy_ids=ids,patch_size=16,positions={str(i):[[0,0],[8,8]] for i in ids},
        input_sha256=hashes,feature_table_sha256=sha256(extraction/'alloy_features.csv'))
    (extraction/'extraction.json').write_text(json.dumps(meta))
    pd.DataFrame(manifest).to_csv(tmp_path/'manifest.csv',index=False)
    pd.DataFrame(rows).to_csv(tmp_path/'labels.csv',index=False)
    cfg=yaml.safe_load((training.ROOT/'configs/mechanics_cnn.yaml').read_text())
    cfg['mechanics_cnn'].update(extraction=str(extraction),manifest=str(tmp_path/'manifest.csv'),
        labels=str(tmp_path/'labels.csv'),output_root=str(tmp_path/'output'),expected_alloys=6,
        software_fixture=True,patch_size=16,seeds=[7],inner_folds=2,max_epochs=1,
        groups_per_epoch=2,batch_groups=2,group_size=2,device='cpu')
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(cfg));training.prepare(p)
    return p,tmp_path/'output'


def test_nested_training_isolates_alloy_and_fold_normalization(fixture,monkeypatch):
    path,root=fixture;_,plan=training.load(path);windows=training.Windows(plan)
    calls=[];original=training.fit
    def observed(windows,ids,*args,**kwargs):
        calls.append(list(ids));return original(windows,ids,*args,**kwargs)
    monkeypatch.setattr(training,'fit',observed)
    for i in plan['alloy_ids']:
        calls.clear();training.train_outer(root,plan,windows,i,7)
        assert all(i not in ids and 3 not in ids for ids in calls)
        row=json.loads((root/'cv'/f'alloy_{i:02d}_seed_7'/'result.json').read_text())
        assert np.allclose(row['target_mean'],windows.labels(row['train_ids']).mean(0))
        for part in row['inner']:
            assert not set(part['train_ids']) & set(part['validation_ids'])
    training.collect(root,plan)
    summary=json.loads((root/'summary.json').read_text())
    assert summary['approved_targets']==[False,False] and not summary['training_authorized']
    with pytest.raises(ValueError,match='review'):training.refit(root,plan,windows,None)
    with pytest.raises(FileExistsError):training.prepare(path)


def test_plan_detects_data_changes(fixture):
    path,root=fixture;plan=json.loads((root/'plan.json').read_text())
    p=Path(plan['settings']['labels']);p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):training.load(path)


def test_reviewed_cnn_feedback_scope_preserves_predictor_training_ids(tmp_path):
    from types import SimpleNamespace
    from ebsd_diffusion.mechanics_runtime import MechanicsRuntime
    labels = tmp_path/'labels.csv'; labels.write_text('alloy_id\n1\n')
    img1=tmp_path/'1.png'; img1.write_bytes(b'one')
    img2=tmp_path/'2.png'; img2.write_bytes(b'two')
    runtime=object.__new__(MechanicsRuntime)
    runtime.is_cnn=True
    runtime.proxy=SimpleNamespace(artifact={
        'training_alloy_ids':[1], 'labels_sha256':sha256(labels),
        'input_sha256':{str(img1.resolve()):sha256(img1), str(img2.resolve()):sha256(img2)}})
    dataset=SimpleNamespace(alloy_ids=[1,2],rows=[{'image_path':str(img1)},{'image_path':str(img2)}])
    runtime.decision={}
    with pytest.raises(ValueError,match='folds differ'): runtime.bind_data(dataset,labels)
    runtime.decision={'feedback_alloy_ids':[1,2],'excluded_alloy_id':3}
    runtime.bind_data(dataset,labels)
    runtime.decision={'feedback_alloy_ids':[1,2,3],'excluded_alloy_id':3}
    with pytest.raises(ValueError,match='scope differs'): runtime.bind_data(dataset,labels)
    runtime.decision={'feedback_alloy_ids':[1,2],'excluded_alloy_id':3}
    img2.write_bytes(b'changed')
    with pytest.raises(ValueError,match='images differ'): runtime.bind_data(dataset,labels)
