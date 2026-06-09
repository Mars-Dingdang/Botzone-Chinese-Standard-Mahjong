#!/usr/bin/env python3
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mahjong_agent.evaluation import evaluate, evaluate_duplicate
from mahjong_agent.policies import HeuristicPolicy, RandomPolicy

def main():
    p=argparse.ArgumentParser(); p.add_argument('--games',type=int,default=400); p.add_argument('--duplicate',action='store_true'); p.add_argument('--model',default=''); p.add_argument('--seed',type=int,default=2026); p.add_argument('--policy-name',default='model'); a=p.parse_args(); policy=HeuristicPolicy()
    if a.model:
        import torch
        from mahjong_agent.models.hybrid_transformer import HybridTransformer
        from mahjong_agent.policies.model import ModelPolicy
        from mahjong_agent.training.checkpoint import load_checkpoint
        model=HybridTransformer(); load_checkpoint(a.model,model); model.to('cuda' if torch.cuda.is_available() else 'cpu'); policy=ModelPolicy(model)
    result=evaluate_duplicate(policy,HeuristicPolicy(),walls=max(1,a.games//4),seed=a.seed,policy_a_name=a.policy_name,policy_b_name='heuristic') if a.duplicate else evaluate([policy,HeuristicPolicy(),RandomPolicy(2),RandomPolicy(3)],games=a.games,seed=a.seed)
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
