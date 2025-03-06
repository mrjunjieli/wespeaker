# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tableprint as tp

import torch
import torchnet as tnt
from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug
import numpy as np

EPOCH=51
mse =torch.nn.MSELoss()

def run_epoch(dataloader, epoch_iter, model, criterion, optimizer, scheduler,
              margin_scheduler, epoch, logger, scaler, device, configs):
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        utts = batch['key']
        targets = batch['label']
        targets = targets.long().to(device)  # (B)

        if frontend_type == 'fbank':
            features = batch['feat']  # (B,T,F)
            features = features.float().to(device)
        else:  # 's3prl'
            wavs = batch['wav']  # (B,1,W)
            wavs = wavs.squeeze(1).float().to(device)  # (B,W)
            wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                wavs.shape[0]).to(device)  # (B)
            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                features, _ = model.module.frontend(wavs, wavs_len)

        with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
            # apply cmvn
            if configs['dataset_args'].get('cmvn', True):
                features = apply_cmvn(
                    features, **configs['dataset_args'].get('cmvn_args', {}))
            # spec augmentation
            if configs['dataset_args'].get('spec_aug', False):
                features = spec_aug(features,
                                    **configs['dataset_args']['spec_aug_args'])

            outputs = model(features)  # (embed_a,embed_b) in most cases
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            
            if configs['model']:
                if i==0:
                    model.module.uttemb={}
                
                if epoch>EPOCH:
                    spk_cluster_batch = torch.cat([torch.tensor(model.module.cluster_for_each_spk[label.item()]).unsqueeze(0) for label in targets],dim=0)
                    spk_cluster_batch = spk_cluster_batch.float().to(device)
                    
                    sigma = outputs[0]
                    var = sigma**0.5
                    ground_var = torch.abs(embeds- spk_cluster_batch)

                    kld = torch.log(var / ground_var) + (ground_var**2 + (spk_cluster_batch - embeds) ** 2) / (2 * var**2) - 0.5
                    kld_loss = torch.mean(kld)

            outputs = model.module.projection(embeds, targets)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)
            
            if epoch>EPOCH:
                # alpha = (100-epoch)/50 if epoch <100 else 0
                alpha = (100-epoch)/50
                loss = loss + 0.5*(1-alpha)* kld_loss

        # loss, acc
        loss_meter.add(loss.item())
        acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())

        # updata the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # log
        if (i + 1) % configs['log_batch_interval'] == 0:
            logger.info(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_meter.value()[0]),
                       width=10,
                       style='grid'))
        if i==10:
            break
        if (i + 1) == epoch_iter:
            break

    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))



def eval_epoch(dataloader, epoch_iter, model, criterion, optimizer, scheduler,
              margin_scheduler, epoch, logger, scaler, device, configs):
    model.eval()
    with torch.no_grad():
        frontend_type = configs['dataset_args'].get('frontend', 'fbank')
        for i, batch in enumerate(dataloader):
            cur_iter = (epoch - 1) * epoch_iter + i
            scheduler.step(cur_iter)
            margin_scheduler.step(cur_iter)

            utts = batch['key']
            targets = batch['label']
            targets = targets.long().to(device)  # (B)

            if frontend_type == 'fbank':
                features = batch['feat']  # (B,T,F)
                features = features.float().to(device)
            else:  # 's3prl'
                wavs = batch['wav']  # (B,1,W)
                wavs = wavs.squeeze(1).float().to(device)  # (B,W)
                wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                    wavs.shape[0]).to(device)  # (B)
                with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                    features, _ = model.module.frontend(wavs, wavs_len)

            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                # apply cmvn
                if configs['dataset_args'].get('cmvn', True):
                    features = apply_cmvn(
                        features, **configs['dataset_args'].get('cmvn_args', {}))
                # spec augmentation
                if configs['dataset_args'].get('spec_aug', False):
                    features = spec_aug(features,
                                        **configs['dataset_args']['spec_aug_args'])
                outputs = model(features)  # (embed_a,embed_b) in most cases
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                
                if configs['model']:
                    if epoch>=EPOCH:
                        if i==0:
                            model.module.uttemb={}
                        for idx, label in enumerate(targets):
                            if label.item() not in model.module.uttemb:
                                model.module.uttemb[label.item()] = [embeds[idx].detach().cpu()]
                            else: 
                                model.module.uttemb[label.item()].append(embeds[idx].detach().cpu())

            if (i + 1) == epoch_iter:
                break
        if epoch >= EPOCH:
            if configs['model']:
                for key, value in model.module.uttemb.items():
                    model.module.cluster_for_each_spk[key]=np.mean(value,axis=0)