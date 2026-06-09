#!/usr/bin/env python3
"""Behavior cloning with DDP, AMP, validation, scheduling and early stopping."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mahjong_agent.engine.actions import ActionType
from mahjong_agent.models.hybrid_transformer import HybridTransformer
from mahjong_agent.training.checkpoint import early_stopping_state, load_checkpoint, save_checkpoint
from mahjong_agent.training.dataset import collate_records, has_tensor_cache, iter_records, iter_tensor_batches, parquet_shard_plan, tensor_shard_plan

def batches(it,n,drop_last=False):
    b=[]
    for x in it:
        b.append(x)
        if len(b)==n: yield b; b=[]
    if b and not drop_last: yield b

def epoch(model,opt,data,bs,device,scaler,train,max_steps,prebatched=False):
    model.train(train); correct=total=steps=0; loss_sum=0.; ctx=torch.enable_grad if train else torch.no_grad
    type_correct=dict((kind.name,0) for kind in ActionType); type_total=dict((kind.name,0) for kind in ActionType)
    for records in (data if prebatched else batches(data,bs,drop_last=train)):
        f,a,m,y=records if prebatched else collate_records(records,torch); f,a,m,y=f.to(device),a.to(device),m.to(device),y.to(device)
        with ctx():
            with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
                out=model(f,a,m); loss=torch.nn.functional.cross_entropy(out['logits'],y)+0.0*(out['value'].sum()+out['aux'].sum())
            if train:
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); scaler.step(opt); scaler.update()
        predicted=out['logits'].argmax(1); matches=predicted==y
        correct+=int(matches.sum()); total+=len(y); loss_sum+=float(loss)*len(y); steps+=1
        chosen_types=torch.round(a[torch.arange(len(y),device=device),y,0]*(len(ActionType)-1)).long()
        for kind in ActionType:
            selected=chosen_types==int(kind); count=int(selected.sum())
            type_total[kind.name]+=count
            type_correct[kind.name]+=int((matches & selected).sum())
        if max_steps and steps>=max_steps: break
    result={'loss':loss_sum/max(total,1),'accuracy':correct/float(max(total,1)),'samples':total,'steps':steps}
    result['accuracy_by_action']=dict((name,type_correct[name]/float(max(1,type_total[name]))) for name in type_total)
    result['samples_by_action']=type_total
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='artifacts/official_bc'); p.add_argument('--output',default='artifacts/bc_model.pt'); p.add_argument('--epochs',type=int,default=15); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--resume',default=''); p.add_argument('--max-steps',type=int,default=0); p.add_argument('--patience',type=int,default=3); p.add_argument('--lr-patience',type=int,default=1); x=p.parse_args()
    dist=int(os.environ.get('WORLD_SIZE','1'))>1; rank=int(os.environ.get('RANK','0')); local=int(os.environ.get('LOCAL_RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
    if dist: torch.distributed.init_process_group('nccl'); torch.cuda.set_device(local)
    dev=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu'); model=HybridTransformer().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=x.lr); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=0.5,patience=x.lr_patience); start=0; best=-1.; stale=0
    if x.resume:
        resumed=load_checkpoint(x.resume,model,opt,scheduler=scheduler); start=int(resumed.get('epoch',0)); best=float(resumed.get('best_accuracy',resumed.get('val',{}).get('accuracy',-1.))); stale=int(resumed.get('stale_epochs',0))
    if dist: model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local])
    scaler=torch.cuda.amp.GradScaler(enabled=dev.type=='cuda')
    for e in range(start,x.epochs):
        train_steps=x.max_steps
        tensor_cache=has_tensor_cache(x.data)
        if tensor_cache:
            _, rank_steps=tensor_shard_plan(x.data,'train',world,x.batch_size)
            train_steps=min(rank_steps) if not train_steps else train_steps
            train_data=iter_tensor_batches(x.data,'train',x.batch_size,rank,world,train_steps)
            if rank==0: print('balanced_rank_steps=%r train_steps=%d'%(rank_steps,train_steps),flush=True)
        else:
            if dist and not train_steps:
                _, rank_samples=parquet_shard_plan(x.data,'train',world)
                train_steps=min(rank_samples)//x.batch_size
                if rank==0: print('balanced_rank_samples=%r train_steps=%d'%(rank_samples,train_steps),flush=True)
            train_data=iter_records(x.data,rank,world,'train')
        tr=epoch(model,opt,train_data,x.batch_size,dev,scaler,True,train_steps,tensor_cache)
        if dist: torch.distributed.barrier()
        if rank==0:
            validation_model=model.module if dist else model; val_data=iter_tensor_batches(x.data,'val',x.batch_size,max_steps=x.max_steps) if tensor_cache else iter_records(x.data,split='val'); va=epoch(validation_model,opt,val_data,x.batch_size,dev,scaler,False,x.max_steps,tensor_cache); scheduler.step(va['accuracy'])
            best,stale,improved=early_stopping_state(best,stale,va['accuracy'])
            print('epoch=%d lr=%g train=%r val=%r best=%g stale=%d'%(e+1,opt.param_groups[0]['lr'],tr,va,best,stale),flush=True); saved=model.module if dist else model; meta={'algorithm':'bc','epoch':e+1,'train':tr,'val':va,'best_accuracy':best,'stale_epochs':stale}; save_checkpoint(x.output,saved,opt,meta,scheduler)
            if improved: save_checkpoint(x.output.replace('.pt','.best.pt'),saved,opt,meta,scheduler)
        stop=stale>=x.patience if rank==0 else False
        if dist:
            control=torch.tensor([float(stop),opt.param_groups[0]['lr']],device=dev)
            torch.distributed.broadcast(control,0)
            stop=bool(control[0].item())
            for group in opt.param_groups: group['lr']=float(control[1].item())
        if dist: torch.distributed.barrier()
        if stop:
            if rank==0: print('early_stopping epoch=%d best_accuracy=%g'%(e+1,best),flush=True)
            break
    if dist: torch.distributed.destroy_process_group()
if __name__=='__main__': main()
