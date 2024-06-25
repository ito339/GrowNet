#!/usr/bin/env python
import numpy as np
import pandas as pd
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
import hydra
from sklearn.metrics import matthews_corrcoef, recall_score, f1_score, confusion_matrix
from omegaconf import DictConfig, OmegaConf
#mlflowのFuture Warningを無視
import warnings
warnings.simplefilter('ignore', FutureWarning)
import mlflow
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--feat_d', type=int, required=True)
# parser.add_argument('--hidden_d', type=int, required=True)
# parser.add_argument('--boost_rate', type=float, required=True)
# parser.add_argument('--lr', type=float, required=True)
# parser.add_argument('--num_nets', type=int, required=True)
# parser.add_argument('--data', type=str, required=True)
# parser.add_argument('--tr', type=str, required=True)
# parser.add_argument('--te', type=str, required=True)
# parser.add_argument('--batch_size', type=int, required=True)
# parser.add_argument('--epochs_per_stage', type=int, required=True)
# parser.add_argument('--correct_epoch', type=int ,required=True)
# parser.add_argument('--L2', type=float, required=True)
# parser.add_argument('--sparse', default=False, type=lambda x: (str(x).lower() == 'true'))
# parser.add_argument('--normalization', default=False, type=lambda x: (str(x).lower() == 'true'))
# parser.add_argument('--cv', default=False, type=lambda x: (str(x).lower() == 'true')) 
# parser.add_argument('--model_order',default='second', type=str)
# parser.add_argument('--out_f', type=str, required=True)
# parser.add_argument('--cuda', action='store_true')

# cfg.opt = parser.parse_args()

# if not cfg.opt.cuda:
#     torch.set_num_threads(16)

# prepare the dataset
def get_data(task_num, target, cfg):
    if cfg.opt.data in ['a9a', 'ijcnn1']:
        train = LibSVMData(cfg.opt.tr, cfg.opt.feat_d, cfg.opt.normalization)
        test = LibSVMData(cfg.opt.te, cfg.opt.feat_d, cfg.opt.normalization)
    elif cfg.opt.data == 'covtype':
        train = LibSVMData(cfg.opt.tr, cfg.opt.feat_d,cfg.opt.normalization, 1, 2)
        test = LibSVMData(cfg.opt.te, cfg.opt.feat_d, cfg.opt.normalization, 1, 2)
    elif cfg.opt.data == 'mnist28':
        train = LibSVMData(cfg.opt.tr, cfg.opt.feat_d, cfg.opt.normalization, 2, 8)
        test = LibSVMData(cfg.opt.te, cfg.opt.feat_d, cfg.opt.normalization, 2, 8)
    elif cfg.opt.data == 'higgs':
        train = LibSVMData(cfg.opt.tr, cfg.opt.feat_d,cfg.opt.normalization, 0, 1)
        test = LibSVMData(cfg.opt.te, cfg.opt.feat_d,cfg.opt.normalization, 0, 1)
    elif cfg.opt.data == 'real-sim':
        train = LibSVMDataSp(cfg.opt.tr, cfg.opt.feat_d)
        test = LibSVMDataSp(cfg.opt.te, cfg.opt.feat_d)
    elif cfg.opt.data in ['criteo', 'criteo2', 'Allstate']:
        train = LibCSVData(cfg.opt.tr, cfg.opt.feat_d, 1, 0)
        test = LibCSVData(cfg.opt.te, cfg.opt.feat_d, 1, 0)
    elif cfg.opt.data == 'yahoo.pair':
        train = LibCSVData(cfg.opt.tr, cfg.opt.feat_d)
        test = LibCSVData(cfg.opt.te, cfg.opt.feat_d)
    # csvファイルを読み込み
    elif cfg.opt.data == 'my_csv_data':
        # train = MYCSVData(cfg.opt.tr, cfg.opt.feat_d, label_col=1)
        # test = MYCSVData(cfg.opt.te, cfg.opt.feat_d, label_col=1)
        data = MYCSVData(cfg.opt.tr, task_num, target)
        return data
    else:
        pass
    #     train = []
    #     test = []
    # if cfg.opt.cv:
    #     val = copy.deepcopy(train)

    #     # Split the data from cut point
    #     print('Creating Validation set! \n')
    #     indices = list(range(len(train))) #全データのインデックス
    #     cut = int(len(train)*0.80) #データを80%で分割
    #     np.random.seed(42)
    #     np.random.shuffle(indices)
    #     train_idx = indices[:cut]
    #     test_idx = indices[cut:]

    #     train.feat = train.feat[train_idx]
    #     train.label = train.label[train_idx]
    #     test.feat = val.feat[test_idx]
    #     test.label = val.label[test_idx]

    # #これは本文中に移動させよう．
    # if cfg.opt.normalization:
    #     scaler = MinMaxScaler() #StandardScaler()
    #     # print(cfg.opt.tr)
    #     # print(train.feat)
    #     scaler.fit(train.feat)
    #     train.feat = scaler.transform(train.feat)
    #     test.feat = scaler.transform(test.feat)
    #     if cfg.opt.cv:
            
    #         val.feat = scaler.transform(val.feat)

    # print(f'#Train: {len(train)}, #Val: {len(val)} #Test: {len(test)}')
    # return train, test, val


def get_optim(params, lr, weight_decay):
    optimizer = Adam(params, lr, weight_decay=weight_decay)
    return optimizer

def accuracy(net_ensemble, test_loader, cfg):
    correct = 0
    total = 0
    loss = 0
    for x, y in test_loader:
        if cfg.opt.cuda:
            x, y = x.float().cuda(), y.float().cuda()
        with torch.no_grad():
            middle_feat, out = net_ensemble.forward(x)
        correct += (torch.sum(y[out > 0.] > 0) + torch.sum(y[out < .0] < 0)).item()
        total += y.numel()
    return correct / total

def logloss(net_ensemble, test_loader, cfg):
    loss = 0
    total = 0
    loss_f = nn.BCEWithLogitsLoss() # Binary cross entopy loss with logits, reduction=mean by default
    for x, y in test_loader:
        if cfg.opt.cuda:
            x, y= x.float().cuda(), y.float().cuda().view(-1)
            # x, y= x.float().cuda(), y.long()
            # num_classes = 2
            # y = torch.nn.functional.one_hot(torch.tensor(y), num_classes).cuda().float()
        y = (y + 1) / 2
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1) # ここ.view(-1, 1)を入れるとバグる
        loss += loss_f(out, y)
        total += 1

    return loss / total

# def specificity_score(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
#     return tn / (tn + fp)

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 1:  # 1x1 matrix
        return 1.0 if cm[0][0] == 0 else 0.0  # すべて正しく予測された場合は1、そうでなければ0
    elif cm.size == 4:  # 2x2 matrix
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) != 0 else 0.0
    else:
        raise ValueError("Unexpected confusion matrix shape")

def auc_score(net_ensemble, test_loader, cfg):
    actual = []
    posterior = []
    predict = []
    for x, y in test_loader:
        if cfg.opt.cuda:
            x = x.float().cuda()
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        prob = 1.0 - 1.0 / torch.exp(out)   # Why not using the scores themselve than converting to prob
        # 両者の事後確率から0,1に変換
        for i in range(len(prob)):
            i_prob = prob[i]
            pred = 1 if i_prob > 0.5 else 0
            predict.append(pred)
        prob = prob.cpu().numpy().tolist()
        posterior.extend(prob)
        y = (y + 1) / 2
        actual.extend(y.numpy().tolist())
        actual = [int(x) for x in actual]
    score = auc(actual, predict)
    return score

def F1_measure(net_ensemble, test_loader, cfg):
    actual = []
    posterior = []
    predict = []
    for x, y in test_loader:
        if cfg.opt.cuda:
            x = x.float().cuda()
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        prob = 1.0 - 1.0 / torch.exp(out)   # Why not using the scores themselve than converting to prob
        # 両者の事後確率から0,1に変換
        for i in range(len(prob)):
            i_prob = prob[i]
            pred = 1 if i_prob > 0.5 else 0
            predict.append(pred)
        prob = prob.cpu().numpy().tolist()
        posterior.extend(prob)
        y = (y + 1) / 2
        actual.extend(y.numpy().tolist())
        actual = [int(x) for x in actual]
    f1 = f1_score(actual, predict)
    recall = recall_score(actual, predict)
    specificity = specificity_score(actual, predict)
    return f1, recall, specificity

def mcc_score(net_ensemble, test_loader, cfg):
    actual = []
    posterior = []
    predict = []
    for x, y in test_loader:
        if cfg.opt.cuda:
            x = x.float().cuda()
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        prob = 1.0 - 1.0 / torch.exp(out)   # Why not using the scores themselve than converting to prob
        # 両者の事後確率から0,1に変換
        for i in range(len(prob)):
            i_prob = prob[i]
            pred = 1 if i_prob > 0.5 else 0
            predict.append(pred)
        prob = prob.cpu().numpy().tolist()
        posterior.extend(prob)
        y = (y + 1) / 2
        actual.extend(y.numpy().tolist())
        actual = [int(x) for x in actual]
    print("actual", actual)
    # print("posterior", posterior)
    print("predict", predict)
    score = matthews_corrcoef(actual, predict)
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

@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg)) #hydraに必要な設定
    task_num = cfg.opt.task_number
    results = []
    print("="*10, "task ", str(task_num), "="*10)
    # print("hidden_d, lr, L2, num_nets, hidden_d")
    # print(cfg.opt.hidden_d, cfg.opt.lr, cfg.opt.L2, cfg.opt.num_nets, cfg.opt.hidden_d)

    #データ読み込み
    data = get_data(task_num, cfg.opt.target,cfg)
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for fold, (train_idx, test_idx) in enumerate(skf.split(data.feat, data.label)):
        print(f"Fold {fold+1}/5")
        train = copy.deepcopy(data)
        val = copy.deepcopy(data)
        test = copy.deepcopy(data)
        
        # Split the data from cut point
        print('Creating Validation set! \n')
        indices = list(range(len(train_idx))) #訓練データのインデックス
        cut = int(len(train_idx)*0.75) #データを80%で分割
        np.random.seed(42)
        np.random.shuffle(indices)
        sub_train_idx = indices[:cut]
        val_idx = indices[cut:]

        train.feat = train.feat[sub_train_idx]
        train.label = train.label[sub_train_idx]
        val.feat = val.feat[val_idx]
        val.label = val.label[val_idx]
        test.feat = test.feat[test_idx]
        test.label = test.label[test_idx]
        print(np.shape(val.feat))
        if cfg.opt.normalization:
            scaler = MinMaxScaler() #StandardScaler()
            # print(cfg.opt.tr)
            # print(train.feat)
            scaler.fit(train.feat)
            train.feat = scaler.transform(train.feat)
            test.feat = scaler.transform(test.feat)
            if cfg.opt.cv:
                val.feat = scaler.transform(val.feat)

        print(f'#Train: {len(train)}, #Val: {len(val)} #Test: {len(test)}')
        print(cfg.opt.data + ' training and test datasets are loaded!')
        #学習データとテストデータをDataLoaderで分割
        #num_workersは並列処理の数
        train_loader = DataLoader(train, cfg.opt.batch_size, shuffle = True, drop_last=True, num_workers=2)
        test_loader = DataLoader(test, cfg.opt.batch_size, shuffle=False, drop_last=True, num_workers=2)
        if cfg.opt.cv:
            val_loader = DataLoader(val, cfg.opt.batch_size, shuffle=True, drop_last=True, num_workers=2)
        # For CV use
        best_stage = cfg.opt.num_nets-1
        best_score = -1
        val_score = best_score
        best_f1_score = 0
        best_mcc_score = 0
        best_AUC_score = 0
        best_recall_score = 0
        best_spec_score = 0

        #データ数の多い方を予測した結果（データの偏り）
        c0 = init_gbnn(train)
        #動的ニューラルネットワーク（Neural Boostingのこと）
        net_ensemble = DynamicNet(c0, cfg.opt.boost_rate)
        
        # lossの重みを追加するためにtrain.csvを読み込み
        df = pd.read_csv(cfg.opt.tr,encoding='cp932')
        df_label = df.iloc[1]
        # Numpyの整数型をPyTorchのテンソルに変換し、それをGPUメモリ上に移動する
        count_0 = torch.tensor(df_label==0, dtype=torch.float32).sum().cuda()
        count_1 = torch.tensor(df_label==1, dtype=torch.float32).sum().cuda()
        loss_weights = torch.tensor([count_1, count_0]).cuda()
        loss_f1 = nn.MSELoss(reduction='none')
        loss_f2 = nn.BCEWithLogitsLoss(reduction='none',weight=loss_weights)
        loss_models = torch.zeros((cfg.opt.num_nets, 3))

        all_ensm_losses = []
        all_ensm_losses_te = []
        all_mdl_losses = []
        dynamic_br = []
        
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment(cfg.mlflow.runname)
        with mlflow.start_run():
            mlflow.log_param('hidden_d', cfg.opt.hidden_d) 
            mlflow.log_param('num_nets', cfg.opt.num_nets)
            mlflow.log_param('batch size', cfg.opt.batch_size)
            for stage in range(cfg.opt.num_nets):
                t0 = time.time()
                #### Higgs 100K, 1M , 10M experiment: Subsampling the data each model training time ############
                indices = list(range(len(train)))
                split = 1000000
                indices = sklearn.utils.shuffle(indices, random_state=41)
                train_idx = indices[:split]
                train_sampler = SubsetRandomSampler(train_idx)
                train_loader = DataLoader(train, cfg.opt.batch_size, sampler = train_sampler, drop_last=True, num_workers=2)
                ################################################################################################

                #MLP_2HL：2つの隠れ層を持つ．1つ目の隠れ層と2つ目の隠れ層の間に活性化関数が存在
                model = MLP_2HL.get_model(stage, cfg.opt)  # Initialize the model_k: f_k(x), multilayer perception v2
                if cfg.opt.cuda:
                    #gpuメモリに転送している
                    model.cuda()

                # 最適化アルゴリズム=Adam
                optimizer = get_optim(model.parameters(), cfg.opt.lr, cfg.opt.L2)
                net_ensemble.to_train() # Set the models in ensemble net to train mode

                stage_mdlloss = []
                for epoch in tqdm(range(cfg.opt.epochs_per_stage)):
                    for i, (x, y) in enumerate(train_loader):
                        if cfg.opt.cuda:
                            # print("x",x.shape)
                            # print("y",y.shape)
                            #x, yをそれぞれgpuメモリ上に転送
                            x, y= x.float().cuda(), y.long().cuda().view(-1) # view(-1, 1)があるとバグる？
                        #順伝播計算をしている．middle_featが中間の特徴量，outが最終出力
                        middle_feat, out = net_ensemble.forward(x) #学習中はout=init_gbnnを出力
                        # print("middle_feat", middle_feat)
                        # print("out", out)
                        out = torch.as_tensor(out, dtype=torch.float32).cuda()
                        # print("torch_tensor_out", out)
                        #層の深さによって計算方法が異なる．重みの更新
                        if cfg.opt.model_order=='first':
                            grad_direction = y / (1.0 + torch.exp(y * out))
                        else:
                            h = 1/((1+torch.exp(y*out))*(1+torch.exp(-y*out))).float()
                            grad_direction = y * (1.0 + torch.exp(-y * out))
                            out = torch.as_tensor(out)
                            nwtn_weights = (torch.exp(out) + torch.exp(-out)).abs()
                        _, out = model(x.float(), middle_feat)
                        out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1).float()
                        #lossを計算
                        loss = loss_f1(net_ensemble.boost_rate*out.squeeze(), grad_direction) # T
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
                        cfg.opt.lr /= 2
                        cfg.opt.L2 /= 2
                    optimizer = get_optim(net_ensemble.parameters(), cfg.opt.lr / lr_scaler, cfg.opt.L2)
                    
                    for _ in range(cfg.opt.correct_epoch):
                        for i, (x, y) in enumerate(train_loader):
                            if cfg.opt.cuda:
                                x, y = x.float().cuda(), y.float().cuda().view(-1)
                            
                            #モデルにxを入力
                            _, out = net_ensemble.forward_grad(x)
                            out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1)
                            y = (y + 1.0) / 2.0
                            # print("out.shape", out.shape)
                            # print("out", out)
                            # print("y.shape", y.shape)
                            # print("y", y)
                            loss = loss_f2(out, y).mean() 
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            stage_loss.append(loss.item())

                #テストデータセットを用いてアンサンブルモデルのロスを計算
                sl_te = logloss(net_ensemble, test_loader, cfg)
                # Store dynamic boost rate
                dynamic_br.append(net_ensemble.boost_rate.item())
                # store model
                net_ensemble.to_file(cfg.opt.out_f)
                # データを読み出して，動的アンサンブルモデルを維持
                net_ensemble = DynamicNet.from_file(cfg.opt.out_f, lambda stage: MLP_2HL.get_model(stage, cfg.opt))

                #1ステージに要した時間を計算
                elapsed_tr = time.time()-t0
                sl = 0
                if stage_loss != []:
                    sl = np.mean(stage_loss)

            

                all_ensm_losses.append(sl)
                all_ensm_losses_te.append(sl_te)
                all_mdl_losses.append(sml)
                print(f'Stage - {stage}, training time: {elapsed_tr: .1f} sec, boost rate: {net_ensemble.boost_rate: .4f}, Training Loss: {sl: .4f}, Test Loss: {sl_te: .4f}')


                if cfg.opt.cuda:
                    net_ensemble.to_cuda()
                net_ensemble.to_eval() # Set the models in ensemble net to eval mode
                # Train
                print('Acc results from stage := ' + str(stage) + '\n')
                # f1_score
                val_f1_score, val_recall_score, val_spec_score = F1_measure(net_ensemble, val_loader, cfg)
                test_f1_score, test_recall_score, test_spec_score = F1_measure(net_ensemble, test_loader, cfg)
                print(f'Stage: {stage}, F1_score@Val: {val_f1_score:.4f}, F1_score@Test: {test_f1_score:.4f}')

                # AUC score
                val_AUC_score = auc_score(net_ensemble, val_loader, cfg) 
                test_AUC_score = auc_score(net_ensemble, test_loader, cfg)
                print(f'Stage: {stage}, AUC@Val: {val_AUC_score:.4f}, AUC@Test: {test_AUC_score:.4f}')
                
                # mcc
                val_mcc_score = mcc_score(net_ensemble, val_loader, cfg)
                test_mcc_score = mcc_score(net_ensemble, test_loader, cfg)
                print(f'Stage: {stage}, mcc_score@Val: {val_mcc_score:.4f}, mcc_score@Test: {test_mcc_score:.4f}')
                
                # Boosting回数の検証
                if cfg.opt.cv:    
                    val_score = val_mcc_score
                    if val_score >= best_score:
                        best_score = val_score
                        best_stage = stage
                        #評価指標
                        best_mcc_score = test_mcc_score
                        best_AUC_score = test_AUC_score
                        best_f1_score = test_f1_score
                        best_recall_score = test_recall_score
                        best_spec_score = test_spec_score
                        # mlflow.log_metric('best stage', best_stage)
                        # mlflow.log_metric('f1_score', test_f1_score) # スコア
                        # mlflow.log_metric('AUC_score', test_score) # スコア

                loss_models[stage, 1], loss_models[stage, 2] = val_score, test_mcc_score
            # mlflow.log_metric('stage', stage)
            # mlflow.log_metric('f1_score', test_f1_score) # スコア
            # mlflow.log_metric('AUC_score', test_score) # スコア
        # val_auc, te_auc = loss_models[best_stage, 1], loss_models[best_stage, 2]
        print(f'Best validation stage: {best_stage},  best mcc@Val: {best_score:.4f}, final mcc@Test: {best_mcc_score:.4f}, final f1_score@Test: {best_f1_score:.4f}, final recall_score@Test: {best_recall_score:.4f}, final spec_score@Test: {best_spec_score:.4f}, final AUC_score@Test: {best_AUC_score:.4f}')
        results.append([best_stage+1, best_score, best_mcc_score, best_f1_score, best_recall_score, best_spec_score, best_AUC_score])
        # loss_models = loss_models.detach().cpu().numpy()
        # fname = 'tr_ts_' + cfg.opt.data +'_auc'
        # np.save(fname, loss_models) 
    df = pd.DataFrame(results, columns=['best stage', 'best mcc@val', 'final mcc@Test', 'final f1score@Test', 'final recall score@Test', 'final spec score@Test', 'final AUC@Test'])
    df.to_csv("./results/"+str(task_num)+".csv", encoding='cp932')
        # fname = './results/' + cfg.opt.data + '_cls'
        # np.savez(fname, training_loss=all_ensm_losses, test_loss=all_ensm_losses_te, model_losses=all_mdl_losses, dynamic_boostrate=dynamic_br, params=cfg.opt)


if __name__ == "__main__":
    main()