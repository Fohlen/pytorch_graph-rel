from argparse import Namespace

import lightning.pytorch as pl
import torch.nn
from torch import optim


class LitGraphRel(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, args: Namespace):
        super().__init__()
        self.model = model
        self.args = args

    def training_step(self, batch, batch_idx):
        def get_loss(weight_loss, out, ans):
            out, ans = out.flatten(0, len(out.shape) - 2), ans.flatten(0, len(ans.shape) - 1)
            ls = torch.nn.functional.cross_entropy(out, ans, ignore_index=-1, reduction='none')
            weight = 1.0 - (ans == -1).float()
            weight.masked_fill_(ans > 0, weight_loss)
            ls = (ls * weight).sum() / (weight > 0).sum()
            return ls

        s, inp_sent, inp_pos, dep_fw, dep_bw, ans_ne, ans_rel = batch
        if self.args.arch == '1p':
            out_ne, out_rel = self.model(inp_sent, inp_pos, dep_fw, dep_bw)
            ls_ne, ls_rel = get_loss(self.args.weight_loss, out_ne, ans_ne), \
                get_loss(self.args.weight_loss, out_rel, ans_rel)
            return ls_ne + self.args.weight_alpha * ls_rel

        elif self.args.arch == '2p':
            out_ne1p, out_rel1p, out_ne2p, out_rel2p = self.model(inp_sent, inp_pos, dep_fw, dep_bw)
            ls_ne1p, ls_rel1p = get_loss(self.args.weight_loss, out_ne1p, ans_ne), \
                get_loss(self.args.weight_loss, out_rel1p, ans_rel)
            ls_ne2p, ls_rel2p = get_loss(self.args.weight_loss, out_ne2p, ans_ne), \
                get_loss(self.args.weight_loss, out_rel2p, ans_rel)
            return (ls_ne1p + ls_ne2p) + self.args.weight_alpha * (ls_rel1p + ls_rel2p)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.lr)
