import logging
import argparse
import torch.optim as optim
from utils.train_args import train_argparse
from utils.utils import save_ckpt,mkdir_exp
import tqdm
from torch.autograd import Variable
import numpy as np
from metrics import LogNLLLoss
import model.MPANet
from metrics.metrics import SigmoidMetric,SamplewiseSigmoidMetric
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.cuda import amp
import torch.backends.cudnn as cudnn
import torch.distributed.launch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import lr_scheduler
from utils.utils import JointTransform2D, ImageToImage2D, Image2D
torch.distributed.init_process_group(backend="nccl")
seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
class Trainer(object):
    def __init__(self,args):
        self.args = args
        if args.crop is not None:
            crop = (args.crop, args.crop)
        else:
            crop = None
        """
            to do: 将对应的数据增强集成到对应的
        """
        self.tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
        self.tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
        self.train_dataset = ImageToImage2D(args.train_dataset, self.tf_train)
        self.val_dataset = ImageToImage2D(args.val_dataset, self.tf_val)
        self.dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valloader = DataLoader(self.val_dataset, 1, shuffle=True)
        self.iou_metric = SigmoidMetric()
        self.niou_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.device = torch.device('cuda')
        self.model = model.MPANet.MPANet(img_size=self.args.imgsize,imgchan =
                                         self.args.imgchan, num_branches= self.args.num_branches)
        self.model.to(self.device)
        self.criterion = LogNLLLoss()
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr = args.learning_rate,weight_decay=1e-5)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,T_0=40,T_mult=2,verbose=False)
        self.best_iou = 0
        self.best_niou = 0
    def training(self,epoch):
        tbar = tqdm(self.dataloader)
        self.model.train()
        epoch_running_loss = 0.0
        for batch_idx, (X_batch, y_batch, *rest) in tbar:
            X_batch = Variable(X_batch.to(device='cuda'))
            y_batch = Variable(y_batch.to(device='cuda'))
            output = self.model(X_batch)
            loss = self.criterion(output, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(epoch+batch_idx/len(self.dataloader))
    def testing(self,epoch):
        tbar = tqdm(self.valloader)
        self.model.eval()
        for batch_idx, (X_batch, y_batch, *rest) in tqdm:
            X_batch = Variable(X_batch.to(device='cuda'))
            y_batch = Variable(y_batch.to(device='cuda'))
            y_out = self.model(X_batch)
            eval_loss = self.criterion(y_out, y_batch)
            tmp2 = y_batch.detach().cpu().numpy()
            tmp = y_out.detach().cpu().numpy()
            middle_tmp = y_out[0, 1, :, :].reshape(1, 1, self.args.imgsize, self.args.imgsize)
            middle_tmp2 = y_batch.reshape(1, 1, self.args.imgsize, self.args.imgsize)
            self.iou_metric.update(middle_tmp, middle_tmp2)
            self.niou_metric.update(middle_tmp, middle_tmp2)
            _, IoU = self.iou_metric.get()
            _, nIoU = self.niou_metric.get()
            del X_batch, y_batch, tmp, tmp2, y_out
        

def main(args,checkpoint_path = None):
    """
    如何处理这个逻辑的关系的根据IoU或者nIoU的最大值保存对应的pth,断点重连应该放在一个较大的部分
    用来接续连接。
    """
    trainer = Trainer(args)
    start_epoch = 0
    ckpt_path = mkdir_exp('ckpt')
    if (args.RESUME):
        path_checkpoint = checkpoint_path
        checkpoint = torch.load(path_checkpoint)
        trainer.model.load_state_dict(checkpoint['net'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        trainer.scheduler.load_state_dict(checkpoint['lr_schedule'])
    for epoch in range(start_epoch,args.epoch):
        trainer.training(epoch)
        if(epoch % trainer.args.ckpt_freq == 0):
            save_ckpt(ckpt_path)


if __name__ =='__main__':
    args = train_argparse()
    args = args.parse_args()
    QAQ = Trainer(args)




