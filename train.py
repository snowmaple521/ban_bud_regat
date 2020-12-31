import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch.autograd import Variable


# 计算loss值
def instance_bce_with_logits(logits, labels):

    #:param logits: 预测值
    #:param labels: 真实值
    #:return: 损失值
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


# 计算分数
def compute_score_with_logits(logits, labels):

    # :param logits: 预测值
    # :param labels: 真实值
    # :return: 分数

    logits = torch.max(logits, 1)[1].data
    one_hots = torch.zeros(*labels.size()).cuda()
    # scatter() 和 scatter_() 的作用是一样的，只不过 scatter() 不会直接修改原来的 Tensor，而 scatter_() 会
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output):

    # :param model: 模型
    # :param train_loader:训练集
    # :param eval_loader: 验证集
    # :param num_epochs: 迭代次数
    # :param output: 输出文件位置
    # :return:

    utils.create_dir(output)  # 创建文件夹
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    # 训练
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        for i, (v, b, q, a) in enumerate(train_loader):
            v = Variable(v).cuda() #[512,36,2048]
            b = Variable(b).cuda() #[512,36,6]
            q = Variable(q).cuda() #[512,14]
            a = Variable(a).cuda() #[512,3129]
            pred = model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()  # 执行单个优化步骤
            optim.zero_grad()  # 梯度请0
            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data.item() * v.size(0)
            train_score += batch_score
        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        eval_score, bound = evaluate(model, eval_loader)


        logger.write('epoch %d, time:%.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss:%.2f,score:%.2f' % (total_loss, train_score))
        logger.write('\teval score:%.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    for v, b, q, a in iter(dataloader):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred = model(v, b, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    model.train()
    return score, upper_bound
