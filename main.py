import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import train
import base_model
from dataset import Dictionary, VQAFeatureDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--num_hid',type=int,default=1024)
    parser.add_argument('--model',type=str,default='baseline0_newatt')
    parser.add_argument('--output',type=str,default='saved_model/exp1')
    parser.add_argument('--batch_size',type=int,default=512)
    parser.add_argument('--seed',type=int,default=1111,help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dictionary = Dictionary.load_from_file('data/glove/dictionary.pkl') #word2idx, idx2word
    train_dset = VQAFeatureDataset('train',dictionary) #443757
    eval_dset = VQAFeatureDataset('val',dictionary) #214354
    batch_size = args.batch_size

    constructor  = 'build_%s' % args.model
    model = getattr(base_model,constructor)(train_dset,args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove/glove6b_init_300d.npy')  #init embeding
    model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset,batch_size,shuffle=True,num_workers=1)
    eval_loader = DataLoader(eval_dset,batch_size,shuffle=True,num_workers=1)

    train(model,train_loader,eval_loader,args.epochs,args.output)
