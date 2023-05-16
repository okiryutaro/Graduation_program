import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
import argparse
import pickle
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import torchvision

from tensorboardX import SummaryWriter

from resnet import *

from preprocessing import *

import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# w = 32
# a = 32
# for X in range(1):
#   print('---------', X  ,'回目', '---------')
w = 2
a = 2
# for i in range(4):
#     count = i + 1
    
#     if count % 4 == 1:
#         w +=1
#         a = 1
#     else:
#         a += 1
        
print("重み:",w,"出力:",a) 

# Training settings
parser = argparse.ArgumentParser(description='DoReFa-Net pytorch implementation')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='resnet_w1a32')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/resnet20_baseline')

parser.add_argument('--cifar', type=int, default=100)

parser.add_argument('--Wbits', type=int, default=w)
parser.add_argument('--Abits', type=int, default=a)

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=1e-4)

parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=75)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--cluster', action='store_true', default=False)

cfg = parser.parse_args(args=[])

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu

def main():
  if cfg.cifar == 10:
    print('training CIFAR-10 !')
    dataset = torchvision.datasets.CIFAR10
  elif cfg.cifar == 100:
    print('training CIFAR-100 !')
    dataset = torchvision.datasets.CIFAR100
  else:
    assert False, 'dataset unknown !'

  print('==> Preparing data ..')
  train_dataset = dataset(root=cfg.data_dir, train=True, download=True,
                          transform=cifar_transform(is_training=True))
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                                              num_workers=cfg.num_workers)

  eval_dataset = dataset(root=cfg.data_dir, train=False, download=True,
                          transform=cifar_transform(is_training=False))
  eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                                            num_workers=cfg.num_workers)

  print('==> Building ResNet..')
  model = resnet20(wbits=cfg.Wbits, abits=cfg.Abits).cuda()

  optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  lr_schedu = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
  criterion = torch.nn.CrossEntropyLoss().cuda()
  summary_writer = SummaryWriter(cfg.log_dir)

  if cfg.pretrain:
    model.load_state_dict(torch.load(cfg.pretrain_dir))

  # Training
  def train(epoch):
    print('\n----------------------------Epoch: %d----------------------------' % epoch)
    model.train()
    correct = 0

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
      outputs = model(inputs.cuda())
      loss = criterion(outputs, targets.cuda())
      _, predicted = torch.max(outputs.data, 1)
      correct += predicted.eq(targets.cuda().data).cuda().sum().item()
      # print(batch_idx)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        step = len(train_loader) * epoch + batch_idx
        duration = time.time() - start_time
        '''
        print('%s epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
              (datetime.now(), epoch, batch_idx, loss.item(),
                cfg.train_batch_size * cfg.log_interval / duration))
        '''

        start_time = time.time()
        summary_writer.add_scalar('cls_loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

    acc = 100. * correct / len(train_dataset)

    return acc,loss.item()

  def test(epoch):
    # pass
    model.eval()
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(eval_loader):
      inputs, targets = inputs.cuda(), targets.cuda()

      outputs = model(inputs)
      loss = criterion(outputs, targets.cuda())
      _, predicted = torch.max(outputs.data, 1)
      correct += predicted.eq(targets.data).cpu().sum().item()

    acc = 100. * correct / len(eval_dataset)
    '''
    print('%s------------------------------------------------------ '
          'Precision@1: %.2f%% \n' % (datetime.now(), acc))
    '''
    summary_writer.add_scalar('Precision@1', acc, global_step=epoch)

    return acc, loss.item()

  result = {}
  result["loss"] = []
  result["acc"] = []
  result["val_loss"] = []
  result["val_acc"] = []

  for epoch in range(cfg.max_epochs):
    acc, loss = train(epoch) #訓練データ
    val_acc, val_loss = test(epoch) #テストデータ
    lr_schedu.step(epoch)
    print('epoch %d, loss: %.4f acc: %.4f val_loss: %.4f val_acc: %.4f'
          % (epoch, loss, acc, val_loss, val_acc))
    result["loss"].append(loss)
    result["acc"].append(acc)
    result["val_loss"].append(val_loss)
    result["val_acc"].append(val_acc)
    torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'checkpoint.t7'))

  summary_writer.close()

  if w == 1:
    if a == 1:
        with open(f'./resnet_cifar100_1-1_mix(2^k).pickle', 'wb') as f:
              pickle.dump(result, f)
    elif a == 2:
        with open(f'./resnet_cifar100_1-2_mix(2^k).pickle', 'wb') as f:
              pickle.dump(result, f)
    elif a == 3:
        with open(f'./resnet_cifar100_1-3_mix(2^k).pickle', 'wb') as f:
              pickle.dump(result, f)
    elif a == 4:
        with open(f'./resnet_cifar100_1-4_mix(2^k).pickle', 'wb') as f:
              pickle.dump(result, f)
  # if w == 2:
  #   if a == 1:
  #       with open(f'./resnet_cifar100_2-1_round(2^k).pickle', 'wb') as f:
  #             pickle.dump(result, f)
  #   elif a == 2:
  #       with open(f'./resnet_cifar100_2-2_round(2^k).pickle', 'wb') as f:
  #             pickle.dump(result, f)
  #   elif a == 3:
  #       with open(f'./resnet_cifar100_2-3_round(2^k).pickle', 'wb') as f:
  #             pickle.dump(result, f)
  #   elif a == 4:
  #       with open(f'./resnet_cifar100_2-4_round(2^k).pickle', 'wb') as f:
  #             pickle.dump(result, f)
  # if w == 3:
  #   if a == 1:
  #       with open(f'./resnet_cifar100_3-1_mix(2^k).pickle', 'wb') as f:
  #             pickle.dump(result, f)
  #   elif a == 2:
  #       with open(f'./resnet_cifar100_3-2_mix(2^k).pickle', 'wb') as f:
  #             pickle.dump(result, f)
  #   elif a == 3:
  #       with open(f'./resnet_cifar100_3-3_mix(2^k).pickle', 'wb') as f:
  #             pickle.dump(result, f)
  #   elif a == 4:
  #       with open(f'./resnet_cifar100_3-4_mix(2^k).pickle', 'wb') as f:
  #             pickle.dump(result, f)
  # if w == 4:
  #   if a == 1:
  #       with open(f'./resnet_cifar100_4-1_mix(2^k).pickle', 'wb') as f:
  #             pickle.dump(result, f)
  #   elif a == 2:
  #       with open(f'./resnet_cifar100_4-2_mix(2^k).pickle', 'wb') as f:
  #             pickle.dump(result, f)
  #   elif a == 3:
  #       with open(f'./resnet_cifar100_4-3_mix(2^k).pickle', 'wb') as f:
  #             pickle.dump(result, f)
  #   elif a == 4:
  #       with open(f'./resnet_cifar100_4-4_mix(2^k).pickle', 'wb') as f:
  #             pickle.dump(result, f)
  

if __name__ == '__main__':
    main()
    print('finish')



