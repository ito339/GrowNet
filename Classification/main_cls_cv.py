#!/usr/bin/env python
import numpy as np
import sklearn
import argparse
import copy
import time
import torch
import torch.nn as nn
from data.sparseloader import DataLoader
from data.data import LibSVMData, LibCSVData, CriteoCSVData, MYCSVData
from data.sparse_data import LibSVMDataSp
from models.mlp import MLP_1HL, MLP_2HL, MLP_3HL
from models.dynamic_net import DynamicNet, ForwardType
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import SGD, Adam
from misc.auc import auc


parser = argparse.ArgumentParser()
parser.add_argument('--feat_d', type=int, required=True)
parser.add_argument('--hidden_d', type=int, required=True)
parser.add_argument('--boost_rate', type=float, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--num_nets', type=int, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--tr', type=str, required=True)
parser.add_argument('--te', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epochs_per_stage', type=int, required=True)
parser.add_argument('--correct_epoch', type=int ,required=True)
parser.add_argument('--L2', type=float, required=True)
parser.add_argument('--sparse', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--normalization', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--cv', default=False, type=lambda x: (str(x).lower() == 'true')) 
parser.add_argument('--model_order',default='second', type=str)
parser.add_argument('--out_f', type=str, required=True)
parser.add_argument('--cuda', action='store_true')

opt = parser.parse_args()

if not opt.cuda:
    torch.set_num_threads(16)

# prepare the dataset
def get_data():
    if opt.data in ['a9a', 'ijcnn1']:
        train = LibSVMData(opt.tr, opt.feat_d, opt.normalization)
        test = LibSVMData(opt.te, opt.feat_d, opt.normalization)
    elif opt.data == 'covtype':
        train = LibSVMData(opt.tr, opt.feat_d,opt.normalization, 1, 2)
        test = LibSVMData(opt.te, opt.feat_d, opt.normalization, 1, 2)
    elif opt.data == 'mnist28':
        train = LibSVMData(opt.tr, opt.feat_d, opt.normalization, 2, 8)
        test = LibSVMData(opt.te, opt.feat_d, opt.normalization, 2, 8)
    elif opt.data == 'higgs':
        train = LibSVMData(opt.tr, opt.feat_d,opt.normalization, 0, 1)
        test = LibSVMData(opt.te, opt.feat_d,opt.normalization, 0, 1)
    elif opt.data == 'real-sim':
        train = LibSVMDataSp(opt.tr, opt.feat_d)
        test = LibSVMDataSp(opt.te, opt.feat_d)
    elif opt.data in ['criteo', 'criteo2', 'Allstate']:
        train = LibCSVData(opt.tr, opt.feat_d, 1, 0)
        test = LibCSVData(opt.te, opt.feat_d, 1, 0)
    elif opt.data == 'yahoo.pair':
        train = LibCSVData(opt.tr, opt.feat_d)
        test = LibCSVData(opt.te, opt.feat_d)
    # csvファイルを読み込み
    elif opt.data == 'my_csv_data':
        train = MYCSVData(opt.tr, opt.feat_d, label_col=1)
        test = MYCSVData(opt.te, opt.feat_d, label_col=1)
    else:
        pass

    val = []
    if opt.cv:
        val = copy.deepcopy(train)

        # Split the data from cut point
        print('Creating Validation set! \n')
        indices = list(range(len(train)))
        cut = int(len(train)*0.95)
        np.random.shuffle(indices)
        train_idx = indices[:cut]
        val_idx = indices[cut:]

        train.feat = train.feat[train_idx]
        train.label = train.label[train_idx]
        val.feat = val.feat[val_idx]
        val.label = val.label[val_idx]

    if opt.normalization:
        scaler = MinMaxScaler() #StandardScaler()
        scaler.fit(train.feat)
        train.feat = scaler.transform(train.feat)
        test.feat = scaler.transform(test.feat)
        if opt.cv:
            val.feat = scaler.transform(val.feat)

    print(f'#Train: {len(train)}, #Val: {len(val)} #Test: {len(test)}')
    return train, test, val


def get_optim(params, lr, weight_decay):
    optimizer = Adam(params, lr, weight_decay=weight_decay)
    return optimizer

def accuracy(net_ensemble, test_loader):
    correct = 0
    total = 0
    loss = 0
    for x, y in test_loader:
        if opt.cuda:
            x, y = x.float().cuda(), y.float().cuda()
        with torch.no_grad():
            middle_feat, out = net_ensemble.forward(x)
        correct += (torch.sum(y[out > 0.] > 0) + torch.sum(y[out < .0] < 0)).item()
        total += y.numel()
    return correct / total

def logloss(net_ensemble, test_loader):
    loss = 0
    total = 0
    loss_f = nn.BCEWithLogitsLoss() # Binary cross entopy loss with logits, reduction=mean by default
    for x, y in test_loader:
        if opt.cuda:
            x, y= x.float().cuda(), y.float().cuda().view(-1, 1)
        y = (y + 1) / 2
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
        loss += loss_f(out, y)
        total += 1

    return loss / total

def auc_score(net_ensemble, test_loader):
    actual = []
    posterior = []
    for x, y in test_loader:
        if opt.cuda:
            x = x.float().cuda()
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        prob = 1.0 - 1.0 / torch.exp(out)   # Why not using the scores themselve than converting to prob
        prob = prob.cpu().numpy().tolist()
        posterior.extend(prob)
        actual.extend(y.numpy().tolist())
    score = auc(actual, posterior)
    return score

def init_gbnn(train):
    positive = negative = 0
    for i in range(len(train)):
        if train[i][1] > 0:
            positive += 1
        else:
            negative += 1
    blind_acc = max(positive, negative) / (positive + negative)
    print(f'Blind accuracy: {blind_acc}')
    #print(f'Blind Logloss: {blind_acc}')
    return float(np.log(positive / negative))

if __name__ == "__main__":

    #データ読み込み
    train, test, val = get_data()
    print(opt.data + ' training and test datasets are loaded!')
    #学習データとテストデータをDataLoaderで分割
    train_loader = DataLoader(train, opt.batch_size, shuffle = True, drop_last=False, num_workers=2)
    test_loader = DataLoader(test, opt.batch_size, shuffle=False, drop_last=False, num_workers=2)
    if opt.cv:
        val_loader = DataLoader(val, opt.batch_size, shuffle=True, drop_last=False, num_workers=2)
    # For CV use
    best_score = 0
    val_score = best_score
    best_stage = opt.num_nets-1

    #データ数の多い方を予測した結果（データの偏り）
    c0 = init_gbnn(train)
    #動的ニューラルネットワーク（Neural Boostingのこと）
    net_ensemble = DynamicNet(c0, opt.boost_rate)
    
    loss_f1 = nn.MSELoss(reduction='none')
    loss_f2 = nn.BCEWithLogitsLoss(reduction='none')
    loss_models = torch.zeros((opt.num_nets, 3))

    all_ensm_losses = []
    all_ensm_losses_te = []
    all_mdl_losses = []
    dynamic_br = []

    for stage in range(opt.num_nets):
        t0 = time.time()
        #### Higgs 100K, 1M , 10M experiment: Subsampling the data each model training time ############
        indices = list(range(len(train)))
        split = 1000000
        indices = sklearn.utils.shuffle(indices, random_state=41)
        train_idx = indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = DataLoader(train, opt.batch_size, sampler = train_sampler, drop_last=True, num_workers=2)
        ################################################################################################

        #MLP_2HL：2つの隠れ層を持つ．1つ目の隠れ層と2つ目の隠れ層の間に活性化関数が存在
        model = MLP_2HL.get_model(stage, opt)  # Initialize the model_k: f_k(x), multilayer perception v2
        if opt.cuda:
            #gpuメモリに転送している
            model.cuda()

        # 最適化アルゴリズム=Adam
        optimizer = get_optim(model.parameters(), opt.lr, opt.L2)
        net_ensemble.to_train() # Set the models in ensemble net to train mode

        stage_mdlloss = []
        for epoch in range(opt.epochs_per_stage):
            for i, (x, y) in enumerate(train_loader):
                if opt.cuda:
                    print("x",x.shape)
                    print("y",y.shape)
                    #x, yをそれぞれgpuメモリ上に転送
                    x, y= x.float().cuda(), y.float().cuda().view(-1, 1)
                #順伝播計算をしている．middle_featが中間の特徴量，outが最終出力
                middle_feat, out = net_ensemble.forward(x)
                out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                #層の深さによって計算方法が異なる．重みの更新
                if opt.model_order=='first':
                    grad_direction = y / (1.0 + torch.exp(y * out))
                else:
                    h = 1/((1+torch.exp(y*out))*(1+torch.exp(-y*out))).float()
                    grad_direction = y * (1.0 + torch.exp(-y * out))
                    out = torch.as_tensor(out)
                    nwtn_weights = (torch.exp(out) + torch.exp(-out)).abs()
                _, out = model(x.float(), middle_feat)
                out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1).float()
                #lossを計算
                loss = loss_f1(net_ensemble.boost_rate*out, grad_direction) # T
                loss = loss*h
                loss = loss.mean()
                model.zero_grad() #モデルの勾配をゼロにリセット
                loss.backward() #逆伝播
                optimizer.step() #パラメータの更新
                stage_mdlloss.append(loss.item()) #訓練中に計算されたlossをリストへ追加

        #モデルを追加している．
        net_ensemble.add(model)
        sml = np.mean(stage_mdlloss)


        stage_loss = []
        lr_scaler = 2
        # fully-corrective step 動的アンサンブル学習
        # 過去のステージのモデル予測誤差を修正するために利用
        if stage != 0:
            # Adjusting corrective step learning rate 15の倍数のときに学習率・L2正則化項が2倍になる．
            if stage % 15 == 0:
                #lr_scaler *= 2
                opt.lr /= 2
                opt.L2 /= 2
            optimizer = get_optim(net_ensemble.parameters(), opt.lr / lr_scaler, opt.L2)
            
            for _ in range(opt.correct_epoch):
                for i, (x, y) in enumerate(train_loader):
                    if opt.cuda:
                        x, y = x.float().cuda(), y.float().cuda().view(-1, 1)
                    
                    #モデルにxを入力
                    _, out = net_ensemble.forward_grad(x)
                    out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                    y = (y + 1.0) / 2.0
                    loss = loss_f2(out, y).mean() 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    stage_loss.append(loss.item())

        #テストデータセットを用いてアンサンブルモデルのロスを計算
        sl_te = logloss(net_ensemble, test_loader)
        # Store dynamic boost rate
        dynamic_br.append(net_ensemble.boost_rate.item())
        # store model
        net_ensemble.to_file(opt.out_f)
        # データを読み出して，動的アンサンブルモデルを維持
        net_ensemble = DynamicNet.from_file(opt.out_f, lambda stage: MLP_2HL.get_model(stage, opt))

        #1ステージに要した時間を計算
        elapsed_tr = time.time()-t0
        sl = 0
        if stage_loss != []:
            sl = np.mean(stage_loss)

        

        all_ensm_losses.append(sl)
        all_ensm_losses_te.append(sl_te)
        all_mdl_losses.append(sml)
        print(f'Stage - {stage}, training time: {elapsed_tr: .1f} sec, boost rate: {net_ensemble.boost_rate: .4f}, Training Loss: {sl: .4f}, Test Loss: {sl_te: .4f}')


        if opt.cuda:
            net_ensemble.to_cuda()
        net_ensemble.to_eval() # Set the models in ensemble net to eval mode

        # Train
        print('Acc results from stage := ' + str(stage) + '\n')
        # AUC
        if opt.cv:
            val_score = auc_score(net_ensemble, val_loader) 
            if val_score > best_score:
                best_score = val_score
                best_stage = stage

        test_score = auc_score(net_ensemble, test_loader)
        print(f'Stage: {stage}, AUC@Val: {val_score:.4f}, AUC@Test: {test_score:.4f}')

        loss_models[stage, 1], loss_models[stage, 2] = val_score, test_score

    val_auc, te_auc = loss_models[best_stage, 1], loss_models[best_stage, 2]
    print(f'Best validation stage: {best_stage},  AUC@Val: {val_auc:.4f}, final AUC@Test: {te_auc:.4f}')

    loss_models = loss_models.detach().cpu().numpy()
    fname = 'tr_ts_' + opt.data +'_auc'
    np.save(fname, loss_models) 

    fname = './results/' + opt.data + '_cls'
    np.savez(fname, training_loss=all_ensm_losses, test_loss=all_ensm_losses_te, model_losses=all_mdl_losses, dynamic_boostrate=dynamic_br, params=opt)

