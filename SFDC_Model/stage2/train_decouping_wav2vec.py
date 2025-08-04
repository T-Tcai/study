import os
import sys
current_file_path = os.path.abspath(__file__)
# 计算项目根目录（/home/xinyue/code/ControlTCEmodel）
project_root = os.path.dirname(os.path.dirname(current_file_path))
# 将根目录添加到Python路径的最前面（确保优先搜索）
sys.path.insert(0, project_root)
import logging
import warnings
from collections import defaultdict
import random
from pathlib import Path
import yaml
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from torch.utils.data import RandomSampler,SequentialSampler
import customAudioDataset as data
from customAudioDataset import collate_fn
from decouping_model_wav2vec import Decouping_model
from stage1.utils import (count_parameters, save_master_checkpoint, save_master_checkpoint_mi,set_seed)
from mi import TemporalMI_Estimator
import numpy as np
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def compute_loss(all_losses, current_epoch, total_epochs):
    # 定义主任务和正则化项
    main_tasks = ["speaker", "emotion", "content", "ws", "we", "wc","f0"]
    regularizations = ["mi_ec", "mi_se", "mi_sc"]
    
    # 动态设置warmup阶段
    warmup_epochs = int(0.3 * total_epochs)
    progress = min(current_epoch / warmup_epochs, 1.0)

    # 主任务固定权重（手动设定）
    main_weights = {
        "speaker": 5.0,   # 重要任务权重高
        "emotion": 5.0,
        "content": 5.0,
        "f0": 2.0,
        "ws": 1.0,        # 次要任务权重低
        "we": 1.0,
        "wc": 1.0
    }
    main_loss = sum(main_weights[name] * all_losses[name] for name in main_tasks)

    # 正则化损失动态加权
    reg_loss = (
        (progress * 1.0) * all_losses["mi_ec"] +        # 互信息 ec
        (progress * 1.0) * all_losses["mi_se"] +        # 互信息 se
        (progress * 1.0) * all_losses["mi_sc"]           # 互信息 sc
    )

    return main_loss + reg_loss

def train_one_step(config,epoch, optimizer,optimizer_mi, model,mi_model, trainloader,writer,scheduler):
    model.train()
    mi_model.train()
    data_length=len(trainloader)
    optimizer.zero_grad()
    for idx,(wave1,wave2,wave3,wave4,wave5,wave6,wave7,list) in enumerate(trainloader):
        anchor = wave1.to(device)
        speaker_pos = wave2.to(device)
        speaker_neg = wave3.to(device)
        emotion_pos = wave4.to(device)
        emotion_neg = wave5.to(device)
        optimizer.zero_grad()  # **清空梯度**
        outputs = model(anchor,speaker_pos,speaker_neg,emotion_pos,emotion_neg,wave6,wave7,list)
        speaker_feat = outputs["speaker_feat"]
        emotion_feat = outputs["emotion_feat"]
        content_feat = outputs["content_feat"]

        loss_ws = outputs["loss_ws"]
        loss_we = outputs["loss_we"]
        loss_wc = outputs["loss_wc"]
        loss_speaker = outputs["loss_speaker"]#*config['model']['λ_sia_s']
        loss_emotion = outputs["loss_emotion"]#*config['model']['λ_sia_e']
        loss_content = outputs["loss_content"]
        loss_emotion_f0 = outputs["loss_emotion_f0"]
        
        speaker_feat_mi = speaker_feat.detach()
        emotion_feat_mi = emotion_feat.detach()
        content_feat_mi = content_feat.detach()
        mi_step = 5
        for i in range (mi_step):
            optimizer_mi.zero_grad()  # 清零梯度
            (loss_sc, loss_se, loss_ec), _ = mi_model(speaker_feat_mi, emotion_feat_mi, content_feat_mi)
            loss_mi = loss_sc + loss_se + loss_ec
            loss_mi.backward()
            log_msg = ( f"Epoch {i+1}\tloss_ec: {loss_ec.item():.6f} /{(i + 1)}\tloss_sc: {loss_sc.item():.6f}/{(i + 1)}\tloss_se: {loss_se.item():.6f} /{(i + 1)}")
            logger.info(log_msg)
            torch.nn.utils.clip_grad_norm_(mi_model.parameters(), max_norm=1.0)
            optimizer_mi.step()
            writer.add_scalar('Train/loss_ec', loss_ec , (epoch-1) * len(trainloader) * mi_step + i * len(trainloader) + idx) 
            writer.add_scalar('Train/loss_se', loss_se , (epoch-1) * len(trainloader) * mi_step + i * len(trainloader) + idx) 
            writer.add_scalar('Train/loss_sc', loss_sc , (epoch-1) * len(trainloader) * mi_step + i * len(trainloader) + idx) 
        mi_model.eval()
        _, (mi_sc_val, mi_se_val, mi_ec_val) = mi_model(speaker_feat, emotion_feat, content_feat)
        
        loss = compute_loss(all_losses={
            "speaker": loss_speaker,
            "emotion": loss_emotion,
            "content": loss_content,
            "f0": loss_emotion_f0,
            "ws": loss_ws,
            "we": loss_we,
            "wc": loss_wc,
            "mi_ec": mi_ec_val,
            "mi_se": mi_se_val,
            "mi_sc": mi_sc_val,
           },current_epoch=epoch, total_epochs=config['common']['max_epoch'])

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


        log_msg = (  
            f"Epoch {epoch} {idx+1}/{data_length}\t  loss: {loss.item():.6f} /{(idx + 1)}\t  mi_ec: {mi_ec_val.item():.6f} /{(idx + 1)}\t mi_sc: {mi_sc_val.item():.6f} /{(idx + 1)}\t mi_se: {mi_se_val.item():.6f} /{(idx + 1)}\t loss_sia_s: {loss_speaker.item():.6f} /{(idx + 1)}\t loss_sia_c: {loss_content.item():.6f} /{(idx + 1)}\tloss_sia_e: {loss_emotion.item():.6f} /{(idx + 1)}\t loss_f0: {loss_emotion_f0.item():.6f} /{(idx + 1)}\t loss_ws: {loss_ws.item():.6f} /{(idx + 1)}\t loss_we: {loss_we.item():.6f} /{(idx + 1)}\t loss_wc: {loss_wc.item():.6f} /{(idx + 1)}\t") 
        logger.info(log_msg) 
    
        writer.add_scalar('Train/loss', loss, (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/mi_ec', mi_ec_val, (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/mi_sc', mi_sc_val, (epoch-1) * len(trainloader) + idx)
        writer.add_scalar('Train/mi_se', mi_se_val, (epoch-1) * len(trainloader) + idx)
        writer.add_scalar('Train/loss_sia_s', loss_speaker , (epoch-1) * len(trainloader) + idx)  
        writer.add_scalar('Train/loss_sia_e', loss_emotion, (epoch-1) * len(trainloader) + idx)  
        writer.add_scalar('Train/loss_sia_c', loss_content, (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/loss_f0', loss_emotion_f0, (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/loss_ws', loss_ws, (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/loss_wc', loss_wc, (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/loss_we', loss_we, (epoch-1) * len(trainloader) + idx) 
    scheduler.step()

    

def test(epoch, model, testloader, config, writer):
    model.eval()
    # # save a sample reconstruction (not cropped)
    input_wav = testloader.dataset.get()[1].unsqueeze(0).to(device)

def train(config):
    """train main function."""
    # remove the logging handler "somebody" added
    logger.handlers.clear()

    # set logger
    file_handler = logging.FileHandler(f"{config['checkpoint']['save_folder']}/train_encodec_bs{config['datasets']['batch_size']}_lr{config['optimization']['lr']}.log")
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
    file_handler.setFormatter(formatter)

    # print to screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    # set seed
    if config['common']['seed'] is not None:
        seed = set_seed(config['common']['seed'])

    # set train dataset

    trainset = data.CustomAudioDataset(config=config,mode='train')
    testset = data.CustomAudioDataset(config=config,mode='test')
    # set encodec model and discriminator model

    model = Decouping_model()
    mi_model = TemporalMI_Estimator()
    logger.info(model)
    logger.info(config)
    logger.info(f"Decoupling Model Parameters: {count_parameters(model)} ")
    logger.info(f"model train mode :{model.training}")

    
    model.cuda(device)
    mi_model.cuda(device)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = config['datasets']['batch_size'],
        sampler = RandomSampler(trainset) ,
        num_workers=config['datasets']['num_workers'],
        shuffle=False, 
        pin_memory=config['datasets']['pin_memory'],
        collate_fn=collate_fn,
        )
    
    # testloader = torch.utils.data.DataLoader(
    #     testset,
    #     batch_size=config['datasets']['batch_size'],
    #     sampler=SequentialSampler(testset), 
    #     shuffle=False, collate_fn=collate_fn,
    #     pin_memory=config['datasets']['pin_memory'])
    
    logger.info(f"There are {len(trainloader)} data to train the Decoupling Model")
    # logger.info(f"There are {len(testloader)} data to test the Decoupling Model")
    
    
    params = [p for p in model.parameters() if p.requires_grad]
    mi_params = [p for p in mi_model.parameters() if p.requires_grad]
    optimizer = optim.Adam([{'params': params, 'lr': float(config['optimization']['lr'])}], betas=(0.9, 0.98), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    optimizer_mi = torch.optim.Adam([{'params': mi_params, 'lr': float(config['optimization']['mi_lr'])}])

    resume_epoch = 0
    if config['checkpoint']['resume']:
        # check the checkpoint_path
        assert config['checkpoint']['checkpoint_path'] != '', "resume path is empty"
        model_checkpoint = torch.load(config['checkpoint']['checkpoint_path'], map_location=device)
        mi_model_checkpoint = torch.load(config['checkpoint']['disc_checkpoint_path'], map_location=device)
        model.load_state_dict(model_checkpoint['model_state_dict'])
        mi_model.load_state_dict(mi_model_checkpoint['model_state_dict'])
        resume_epoch = model_checkpoint['epoch']
        if resume_epoch >= config['common']['max_epoch']:
            raise ValueError(f"resume epoch {resume_epoch} is larger than total epochs {config['common']['max_epoch']}")
        #logger.info(f"load chenckpoint of model and disc_model, resume from {resume_epoch}")

    if config['checkpoint']['resume'] and 'scheduler_state_dict' in model_checkpoint.keys() : 
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        optimizer_mi.load_state_dict(mi_model_checkpoint['optimizer_state_dict'])
        logger.info(f"load optimizer and disc_optimizer state_dict from {resume_epoch}")

    writer = SummaryWriter(log_dir=f"{config['checkpoint']['save_folder']}/runs")  
    logger.info(f'Saving tensorboard logs to {Path(writer.log_dir).resolve()}')
   
    start_epoch = max(1,resume_epoch+1) # start epoch is 1 if not resume
    # instantiate loss balancer
    #test(0, model, disc_model, testloader, config, writer)

    for epoch in range(start_epoch, config['common']['max_epoch']+1):
        train_one_step(config,epoch, optimizer,optimizer_mi, model,mi_model, trainloader,writer,scheduler)
        # if epoch % config['common']['test_interval'] == 0:
        #     test(epoch, model, testloader, config, writer)
        # save checkpoint and epoch
        if epoch % config['common']['save_interval'] == 0:
            save_master_checkpoint(epoch, model, optimizer, scheduler, f"{config['checkpoint']['save_folder']}/bs{config['datasets']['batch_size']}_epoch{epoch}_lr{config['optimization']['lr']}.pt")
            save_master_checkpoint_mi(epoch, mi_model, optimizer_mi, f"{config['checkpoint']['save_folder']}/bs{config['datasets']['batch_size']}_epoch{epoch}_mi_lr{config['optimization']['mi_lr']}.pt")    

def main(config): 
    train(config)  # set single gpu train 


if __name__ == '__main__':
    with open("/home/xinyue/code/TCEmodel/stage2/config/config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    main(config)