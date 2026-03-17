import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cpea import CPEA
from models.backbones import BackBone
from dataloader.samplers import CategoriesSampler
from utils import pprint, ensure_path, Averager, count_acc, compute_confidence_interval, draw_confusion_matrix
from tensorboardX import SummaryWriter



all_true_labels = []  # 存储所有任务的真实标签
all_pred_labels = []  # 存储所有任务的预测标签







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--test_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)  #******************
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.00001)
    # parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_mul', type=float, default=100)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--model_type', type=str, default='small')
    parser.add_argument('--dataset', type=str, default='CIFARFS')
    parser.add_argument('--init_weights', type=str, default='/home/ys/22class/wmy/Tmp/CPEA+ATR-Net/initialization/cifarfs/checkpoint1600.pth')
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--exp', type=str, default='CPEA')
    args = parser.parse_args()
    pprint(vars(args))

    save_path = '-'.join([args.exp, args.dataset, args.model_type,'1shot-60em-10000']) #********************
    args.save_path = osp.join('./results', save_path)
    ensure_path(args.save_path)

    if args.dataset == 'MiniImageNet':
        from dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CIFARFS':
        from dataloader.cifarfs import CIFARFS as Dataset
    elif args.dataset == 'tiered_imagenet':
        from dataloader.tiered_imagenet import TieredImagenet as Dataset
    elif args.dataset == 'fc100':
        from dataloader.fc100 import FC100 as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, 100, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, 500, args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

    model = BackBone(args)
    dense_predict_network = CPEA()

    optimizer = torch.optim.Adam([{'params': model.encoder.parameters()}], lr=args.lr, weight_decay=0.001)
    print('Using {}'.format(args.model_type))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.000001)

    dense_predict_network_optim = torch.optim.Adam(dense_predict_network.parameters(), lr=args.lr * args.lr_mul, weight_decay=0.001)
    dense_predict_network_scheduler = torch.optim.lr_scheduler.StepLR(dense_predict_network_optim, step_size=args.step_size, gamma=args.gamma)
    # dense_predict_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.000001 * args.lr_mul)

    # load pre-trained model (no FC weights)
    model_dict = model.state_dict()
    print(model_dict.keys())
    if args.init_weights is not None:
        pretrained_dict = torch.load(args.init_weights, map_location='cpu')['teacher']
        print(pretrained_dict.keys())
        pretrained_dict = {k.replace('backbone', 'encoder'): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        dense_predict_network = dense_predict_network.cuda()


    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
        torch.save(dict(params=dense_predict_network.state_dict()), osp.join(args.save_path, name + '_dense_predict.pth'))

    # trlog = {}
    # trlog['args'] = vars(args)
    # trlog['train_loss'] = []
    # trlog['val_loss'] = []
    # trlog['train_acc'] = []
    # trlog['val_acc'] = []
    # trlog['max_acc'] = 0.0
    # trlog['max_acc_epoch'] = 0
    #
    # global_count = 0
    # writer = SummaryWriter(comment=args.save_path)
    #
    # for epoch in range(1, args.max_epoch + 1):
    #     # lr_scheduler.step()
    #     # dense_predict_network_scheduler.step()
    #     model.train()
    #     dense_predict_network.train()
    #     tl = Averager()
    #     ta = Averager()
    #
    #     for i, batch in enumerate(train_loader, 1):
    #         # zero gradient
    #         optimizer.zero_grad()
    #         dense_predict_network_optim.zero_grad()
    #
    #         # forward and backward
    #         global_count = global_count + 1
    #         if torch.cuda.is_available():
    #             data, _ = [_.cuda() for _ in batch]
    #         else:
    #             data = batch[0]
    #         p = args.shot * args.way
    #         data_shot, data_query = data[:p], data[p:]
    #         feat_shot, feat_query = model(data_shot, data_query)
    #         # support_attn = support_attn.detach()[:, :, 0] #*****[25,6,197]
    #         # query_attn = query_attn.detach()[:, :, 0] #[75,6,197]
    #
    #         results, _ = dense_predict_network(feat_query, feat_shot, args)
    #         results = torch.cat(results, dim=0)  # Q x S
    #         label = torch.arange(args.way).repeat(args.query).long().to('cuda')
    #
    #         eps = 0.1
    #         one_hot = torch.zeros_like(results).scatter(1, label.view(-1, 1), 1)
    #         one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (args.way - 1)
    #         log_prb = F.log_softmax(results, dim=1)
    #
    #         loss = -(one_hot * log_prb).sum(dim=1)
    #         loss = loss.mean()
    #
    #         acc = count_acc(results.data, label)
    #         writer.add_scalar('data/loss', float(loss), global_count)
    #         writer.add_scalar('data/acc', float(acc), global_count)
    #         print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), acc))
    #
    #         tl.add(loss.item())
    #         ta.add(acc)
    #
    #         loss_total = loss
    #
    #         loss_total.backward()
    #         optimizer.step()
    #         dense_predict_network_optim.step()
    #
    #     lr_scheduler.step()
    #     dense_predict_network_scheduler.step()
    #
    #     tl = tl.item()
    #     ta = ta.item()
    #
    #     model.eval()
    #     dense_predict_network.eval()
    #
    #     vl = Averager()
    #     va = Averager()
    #
    #     print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    #     with torch.no_grad():
    #         for i, batch in enumerate(val_loader, 1):
    #             if torch.cuda.is_available():
    #                 data, _ = [_.cuda() for _ in batch]
    #             else:
    #                 data = batch[0]
    #             p = args.shot * args.test_way
    #             data_shot, data_query = data[:p], data[p:]
    #             feat_shot, feat_query = model(data_shot, data_query)
    #             # feat_shot, support_attn, feat_query, query_attn = model(data_shot, data_query)
    #             # support_attn = support_attn.detach()[:, :, 0]  # *****[25,6,197]
    #             # query_attn = query_attn.detach()[:, :, 0]  # [75,6,197]
    #
    #             results, _ = dense_predict_network(feat_query, feat_shot,  args)  # Q x S
    #
    #             results = [torch.mean(idx, dim=0, keepdim=True) for idx in results]
    #
    #             results = torch.cat(results, dim=0)  # Q x S
    #             label = torch.arange(args.test_way).repeat(args.query).long().to('cuda')
    #
    #             loss = F.cross_entropy(results, label)
    #             acc = count_acc(results.data, label)
    #             vl.add(loss.item())
    #             va.add(acc)
    #
    #     vl = vl.item()
    #     va = va.item()
    #     writer.add_scalar('data/val_loss', float(vl), epoch)
    #     writer.add_scalar('data/val_acc', float(va), epoch)
    #     print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
    #
    #     if va >= trlog['max_acc']:
    #         trlog['max_acc'] = va
    #         trlog['max_acc_epoch'] = epoch
    #         save_model('max_acc')
    #
    #     trlog['train_loss'].append(tl)
    #     trlog['train_acc'].append(ta)
    #     trlog['val_loss'].append(vl)
    #     trlog['val_acc'].append(va)
    #
    #     torch.save(trlog, osp.join(args.save_path, 'trlog'))
    #
    #     save_model('epoch-last')
    #
    # writer.close()

    # Test Phase
    trlog = torch.load(osp.join(args.save_path, 'trlog'))
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, 10000, args.test_way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    test_acc_record = np.zeros((10000,))

    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
    model.eval()

    dense_predict_network.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '_dense_predict.pth'))['params'])
    dense_predict_network.eval()

    ave_acc = Averager()
    label = torch.arange(args.test_way).repeat(args.query)

    # #*********
    # iterations = 1001
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # random_input = torch.randn(25, 3, 224, 224).cuda()
    # random_input2 = torch.randn(75, 3, 224, 224).cuda()
    # for _ in range(50):
    #     _, _ = model(random_input,random_input2)
    # times = torch.zeros(iterations)




    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = args.test_way * args.shot
            data_shot, data_query = data[:k], data[k:]
            #
            # starter.record()


            feat_shot, feat_query = model(data_shot, data_query)
            # support_attn = support_attn.detach()[:, :, 0]  # *****[25,6,197]
            # query_attn = query_attn.detach()[:, :, 0]  # [75,6,197]

            results, _ = dense_predict_network(feat_query, feat_shot,  args)  # Q x S

            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)  # 计算时间
            # print(curr_time)
            # times[i] = curr_time



            results = [torch.mean(idx, dim=0, keepdim=True) for idx in results]
            results = torch.cat(results, dim=0)  # Q x S
            label = torch.arange(args.test_way).repeat(args.query).long().to('cuda')




            acc, pred1, label1 = count_acc(results.data, label)


            all_true_labels.extend(pred1)
            all_pred_labels.extend(label1)


            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            print('batch {}: acc {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

    # mean_time = times.mean().item()
    # print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

    # all_true_labels = np.array(all_true_labels)
    # all_pred_labels = np.array(all_pred_labels)
    # draw_confusion_matrix(label_true=all_true_labels,  # y_gt=[0,5,1,6,3,...]
    #                       label_pred=all_pred_labels,  # y_pred=[0,5,1,6,3,...]
    #                       # label_name=["0", "1", "2", "3", "4","5", "6", "7", "8", "9","10", "11", "12", "13", "14","15", "16", "17", "18", "19"],
    #                       label_name=["0", "1", "2", "3", "4"],
    #                       title="Confusion Matrix on Fer2013",
    #                       pdf_save_path="Confusion_Matrix_on_Fer2013.jpg",
    #                       dpi=500)

    m, pm = compute_confidence_interval(test_acc_record)
    print('Val Best Epoch {}, Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
