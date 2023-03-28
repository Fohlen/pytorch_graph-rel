import argparse
import os
from pprint import pprint

import spacy
import torch
import torch.utils.data
import lightning.pytorch as pl
from tqdm import tqdm

from dataset import DS
from model import GraphRel
from module import LitGraphRel


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path", default='nyt', type=str)
    parser.add_argument("--accelerator", default="cpu", type=str)
    parser.add_argument("--num-workers", default=os.cpu_count(), type=int)
    
    parser.add_argument("--max_len", default=120, type=int)
    parser.add_argument("--num_ne", default=5, type=int)
    parser.add_argument("--num_rel", default=25, type=int)
    
    parser.add_argument("--size_hid", default=256, type=int)
    parser.add_argument("--layer_rnn", default=2, type=int)
    parser.add_argument("--layer_gcn", default=2, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--arch", default='2p', type=str)
    
    parser.add_argument("--size_epoch", default=40, type=int)
    parser.add_argument("--size_batch", default=64, type=int)
    parser.add_argument("--lr", default=8e-4, type=float)
    parser.add_argument("--weight_loss", default=2.0, type=float)
    parser.add_argument("--weight_alpha", default=3.0, type=float)
    
    args = parser.parse_args()

    return args


def eval_dl(model, dl, args):
    ret = {'precision': [0, 0], 'recall': [0, 0], 'f1': 0}
    
    I = 0
    for s, inp_sent, inp_pos, dep_fw, dep_bw, ans_ne, ans_rel in tqdm(dl, ascii=True):
        if args.arch=='1p':
            out_ne, out_rel = model(inp_sent, inp_pos, dep_fw, dep_bw)
        elif args.arch=='2p':
            _, _, out_ne, out_rel = model(inp_sent, inp_pos, dep_fw, dep_bw)
        
        out_ne, out_rel = [torch.argmax(out, dim=-1).data.cpu().numpy() for out in [out_ne, out_rel]]
        for o_ne, o_rel in zip(out_ne, out_rel):
            l = len(dl.dataset.dat[I]['sentence'])+1
            
            ne, pos = {}, -1
            for i in range(l):
                v = o_ne[i]
                if v==4:
                    ne[i] = [i, i]
                    pos = -1
                elif v==1:
                    pos = i
                elif v==2:
                    pass
                elif v==3:
                    if pos!=-1:
                        for p in range(pos, i+1):
                            ne[p] = [pos, i]
                elif v==0:
                    pos = -1
            
            pd = set()
            for i in range(l):
                for j in range(l):
                    if o_rel[i][j]!=0 and i in ne and j in ne:
                        pd.add((ne[i][1], ne[j][1], o_rel[i][j]))
            
            gt = set()
            for ne1, ne2, rel in dl.dataset.dat[I]['label']:
                gt.add((ne1[1], ne2[1], rel))
            
            ret['precision'][0] += len(pd.intersection(gt))
            ret['precision'][1] += len(pd)
            ret['recall'][0] += len(pd.intersection(gt))
            ret['recall'][1] += len(gt)
            
            I += 1
    
    ret['precision'] = ret['precision'][0]/ret['precision'][1] if ret['precision'][1]>0 else 0
    ret['recall'] = ret['recall'][0]/ret['recall'][1] if ret['recall'][1]>0 else 0
    ret['f1'] = 2*ret['precision']*ret['recall']/(ret['precision']+ret['recall']) if (ret['precision']+ret['recall'])>0 else 0
    
    return ret


if __name__ == '__main__':
    args = get_args()
    NLP = spacy.load('en_core_web_lg')
    train_dataset = DS(NLP, args.path, "train", args.max_len)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.size_batch, num_workers=args.num_workers
    )
    # we skip the validation set because this is not correctly defined in GraphREL

    model = GraphRel(
        len(NLP.pipe_labels['tagger']) + 1,
        args.num_ne,
        args.num_rel,
        args.size_hid,
        args.layer_rnn,
        args.layer_gcn,
        args.dropout,
        args.arch
    )
    module = LitGraphRel(model, args)
    trainer = pl.Trainer(
        max_epochs=args.size_epoch,
        accelerator=args.accelerator,
    )
    trainer.fit(model=module, train_dataloaders=train_loader)

    dataset_test = DS(NLP, args.path, "test", args.max_len)
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.size_batch,
        num_workers=args.num_workers
    )
    evaluation = eval_dl(model, test_loader, args)
    pprint(evaluation)
