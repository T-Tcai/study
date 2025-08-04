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
import os
import sys
current_file_path = os.path.abspath(__file__)
# 计算项目根目录（/home/xinyue/code/ControlTCEmodel）
project_root = os.path.dirname(os.path.dirname(current_file_path))
# 将根目录添加到Python路径的最前面（确保优先搜索）
sys.path.insert(0, project_root)
from stage1.losses import disc_loss, total_loss
from stage1.balancer import Balancer
from customAudioDataset import collate_fn
from Generate_wav import Generate_model
from stage1.msstftd import MultiScaleSTFTDiscriminator
from stage1.scheduler import WarmupCosineLrScheduler
from stage1.utils import (count_parameters, save_master_checkpoint, save_master_checkpoint_mi,set_seed)
import numpy as np
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def train_one_step(config,epoch, optimizer,optimizer_disc, model,disc_model,trainloader,writer,scheduler,disc_scheduler,balancer):
    model.train()
    disc_model.train()
    if epoch >= 51:
        for p in model.decoder.parameters():
            p.requires_grad = True
        # 添加解码器参数到优化器（使用当前学习率）
        if epoch == 51:  # 只在第61轮添加一次
            current_lr = optimizer.param_groups[0]['lr']  # 获取当前主参数组的学习率
            optimizer.add_param_group({
                'params': model.decoder.parameters(),
                'lr': current_lr  # 继承当前学习率
            })
    print(f"Epoch {epoch} 参数组数量: {len(optimizer.param_groups)}")
    lrs = [group['lr'] for group in optimizer.param_groups]
    print(f"Epoch {epoch} 学习率: {lrs}")

    data_length=len(trainloader)
    for idx,(wave,wav_numpt) in enumerate(trainloader):
        wave = wave.to(device)
        optimizer.zero_grad()  # **清空梯度**
        waveform_gen, gen_con_loss, gen_spk_loss, gen_emo_loss, gate_balance_loss = model(wave,wav_numpt)
        logits_real, fmap_real = disc_model(wave)
        logits_fake, fmap_fake = disc_model(waveform_gen)
        

            
        losses_g = total_loss(
            fmap_real, 
            logits_fake, 
            fmap_fake, 
            wave, 
            waveform_gen, 
            sample_rate=config['model']['sample_rate'],
        )
    
        losses_g['spk_loss'] = gen_spk_loss
        losses_g['emo_loss'] = gen_emo_loss
        losses_g['con_loss'] = gen_con_loss
        losses_g['gate_balance_loss'] = gate_balance_loss
        
        
        if balancer is not None:
            balancer.backward(losses_g, waveform_gen, retain_graph=True)
            # naive loss summation for metrics below
            loss = sum([l * balancer.weights[k] for k, l in losses_g.items()])
        else:
            # without balancer: loss = 3*l_g + 3*l_feat + (l_t / 10) + l_f
            # loss_g = torch.tensor([0.0], device='cuda', requires_grad=True)
            loss = 3*losses_g['l_g'] + 3*losses_g['l_feat'] + losses_g['l_t']/10 + losses_g['l_f'] 
        loss.backward()
        optimizer.step()

        log_msg = (
            f"Epoch {epoch} {idx+1}/{data_length} | "
            f"loss: {loss.item():.6f} | "
            f"loss_t: {losses_g['l_t'].item():.6f} | "
            f"loss_f: {losses_g['l_f'].item():.6f} | "
            f"loss_g: {losses_g['l_g'].item():.6f} | "
            f"loss_feat: {losses_g['l_feat'].item():.6f} | "
            f"loss_gen_spk: {gen_spk_loss.item():.6f} | "
            f"loss_gen_emo: {gen_emo_loss.item():.6f} | "
            f"loss_gen_con: {gen_con_loss.item():.6f} | "
            f"gate_balance_loss: {gate_balance_loss.item():.6f} | "
            f"lr_G: {optimizer.param_groups[0]['lr']:.6e}|"
            f"lr_D: {optimizer_disc.param_groups[0]['lr']:.6e}|"
        )
        
        optimizer_disc.zero_grad()
        train_discriminator = torch.BoolTensor([config['model']['train_discriminator']
                            and epoch >= config['lr_scheduler']['warmup_epoch'] 
                            and random.random() < float(eval(str(config['model']['train_discriminator'])))]).to(device)
        print(train_discriminator)
        if train_discriminator:
            logits_real, fmap_real = disc_model(wave)
            logits_fake, fmap_fake = disc_model(waveform_gen.detach())
            loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss
            loss_disc.backward() 
            optimizer_disc.step()
            log_msg += f"loss_disc: {loss_disc.item() :.6f}"  
            writer.add_scalar('Train/loss_disc', loss_disc, (epoch-1) * len(trainloader) + idx)
        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        logger.info(log_msg)
        scheduler.step()
        disc_scheduler.step()
        
        writer.add_scalar('Train/loss_g',losses_g['l_g'], (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/loss_f', losses_g['l_f'], (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/loss_t',losses_g['l_t'], (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/loss_feat',losses_g['l_feat'], (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/loss_gen_spk', gen_spk_loss, (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/loss_gen_emo', gen_emo_loss, (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/loss_gen_con', gen_con_loss, (epoch-1) * len(trainloader) + idx)
        writer.add_scalar('Train/gate_balance_loss', gate_balance_loss, (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/loss', loss, (epoch-1) * len(trainloader) + idx)
        

        

def test(epoch, model, trainloader,testloader, config, writer):
    model.eval()
    with torch.no_grad(): 
        input_wav_test = testloader.dataset.get()[1].unsqueeze(0).to(device)

        output_test = model.generate_audio(input_wav_test,input_wav_test,input_wav_test.squeeze())
        print(input_wav_test.shape)
        print(output_test.shape)
        print(input_wav_test.device)
        print(output_test.device)
        sp = Path(config['checkpoint']['save_folder'])
        torchaudio.save(sp/f'GT_test.wav', input_wav_test.squeeze(0).cpu(), config['model']['sample_rate'])
        torchaudio.save(sp/f'Reconstruction_test.wav', output_test.squeeze(0).cpu(), config['model']['sample_rate'])
        
        input_wav1_test = testloader.dataset.get()[1].unsqueeze(0).to(device)
        input_wav2_test = testloader.dataset.get()[1].unsqueeze(0).to(device)
        input_wav3_test = testloader.dataset.get()[1].unsqueeze(0).to(device)
        
        output1_test = model.generate_audio(input_wav1_test,input_wav2_test,input_wav3_test.squeeze())
        sp = Path(config['checkpoint']['save_folder'])
        torchaudio.save(sp/f'tim_test.wav', input_wav1_test.squeeze(0).cpu(), config['model']['sample_rate'])
        torchaudio.save(sp/f'emo_test.wav', input_wav2_test.squeeze(0).cpu(), config['model']['sample_rate'])
        torchaudio.save(sp/f'con_test.wav', input_wav3_test.squeeze(0).cpu(), config['model']['sample_rate'])
        torchaudio.save(sp/f'Converstion_test.wav', output1_test.squeeze(0).cpu(), config['model']['sample_rate'])

        input_wav_train = trainloader.dataset.get()[1].unsqueeze(0).to(device)
        output_train = model.generate_audio(input_wav_train,input_wav_train,input_wav_train.squeeze())
        sp = Path(config['checkpoint']['save_folder'])
        torchaudio.save(sp/f'GT_train.wav', input_wav_train.squeeze(0).cpu(), config['model']['sample_rate'])
        torchaudio.save(sp/f'Reconstruction_train.wav', output_train.squeeze(0).cpu(), config['model']['sample_rate'])
        
        input_wav1_train = trainloader.dataset.get()[1].unsqueeze(0).to(device)
        input_wav2_train = trainloader.dataset.get()[1].unsqueeze(0).to(device)
        input_wav3_train = trainloader.dataset.get()[1].unsqueeze(0).to(device)
        output1_train = model.generate_audio(input_wav1_train, input_wav2_train ,input_wav3_train.squeeze())
        sp = Path(config['checkpoint']['save_folder'])
        torchaudio.save(sp/f'tim_train.wav', input_wav1_train.squeeze(0).cpu(), config['model']['sample_rate'])
        torchaudio.save(sp/f'emo_train.wav', input_wav2_train.squeeze(0).cpu(), config['model']['sample_rate'])
        torchaudio.save(sp/f'con_train.wav', input_wav3_train.squeeze(0).cpu(), config['model']['sample_rate'])
        torchaudio.save(sp/f'Converstion_train.wav', output1_train.squeeze(0).cpu(), config['model']['sample_rate'])
    
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

    model = Generate_model()
    disc_model = MultiScaleSTFTDiscriminator(
        in_channels=config['model']['channels'],
        out_channels=config['model']['channels'],
        filters=config['model']['filters'],
        hop_lengths=config['model']['disc_hop_lengths'],
        win_lengths=config['model']['disc_win_lengths'],
        n_ffts=config['model']['disc_n_ffts'],
    )
    
    logger.info(model)
    logger.info(config)
    logger.info(f"Decoupling Model Parameters: {count_parameters(model)} ")
    logger.info(f"model train mode :{model.training}")
    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数：{params/1e6:.2f} M")

    
    logger.info(model)
    logger.info(config)
    logger.info(f"Decoupling Model Parameters: {count_parameters(model)} ")
    logger.info(f"model train mode :{model.training}")
    model.cuda(device)
    disc_model.cuda(device)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = config['datasets']['batch_size'],
        sampler = RandomSampler(trainset) ,
        num_workers=config['datasets']['num_workers'],
        shuffle=False, 
        pin_memory=config['datasets']['pin_memory'],
        collate_fn=collate_fn,
        )
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config['datasets']['batch_size'],
        sampler=SequentialSampler(testset), 
        shuffle=False, collate_fn=collate_fn,
        pin_memory=config['datasets']['pin_memory'])
    
    logger.info(f"There are {len(trainloader)} data to train the Decoupling Model")
    logger.info(f"There are {len(testloader)} data to test the Decoupling Model")
    
    
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.Adam([{'params': params, 'lr': float(config['optimization']['lr'])}], betas=(0.9, 0.98), weight_decay=1e-4)
    
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    decoder_param_ids = set(id(p) for p in decoder_params)
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in decoder_param_ids]

    # 初始化 optimizer（保持顺序和之前一致）
    optimizer = optim.Adam([
        {'params': other_params, 'lr': float(config['optimization']['lr'])},
        {'params': decoder_params, 'lr': float(config['optimization']['lr'])}
    ], betas=(0.9, 0.98), weight_decay=1e-4)
    
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    optimizer_disc = optim.Adam([{'params':disc_params, 'lr': config['optimization']['disc_lr']}], betas=(0.5, 0.9))
    #退火策略
    scheduler = WarmupCosineLrScheduler(
        optimizer, 
        max_iter=config['common']['max_epoch']*len(trainloader),
        eta_ratio=0.3,  # 提升后期学习率
        warmup_iter=config['lr_scheduler']['warmup_epoch']*len(trainloader),
        warmup_ratio=1e-2,  # 初始LR不再极小
        warmup='linear'  # 换线性warmup，提速
    )

    disc_scheduler = WarmupCosineLrScheduler(
        optimizer_disc, 
        max_iter=config['common']['max_epoch']*len(trainloader),
        eta_ratio=0.3,  # 提升后期学习率
        warmup_iter=config['lr_scheduler']['warmup_epoch']*len(trainloader),
        warmup_ratio=1e-2,  # 初始LR不再极小
        warmup='linear'  # 换线性warmup，提速
    )


    balancer = Balancer(dict(config['balancer']['weights'])) if hasattr(config, 'balancer') else None
    if balancer:
        logger.info(f'Loss balancer with weights {balancer.weights} instantiated')
    
    
    resume_epoch = 0
    if config['checkpoint']['resume']:
        # check the checkpoint_path
        assert config['checkpoint']['checkpoint_path'] != '', "resume path is empty"
        model_checkpoint = torch.load(config['checkpoint']['checkpoint_path'], map_location=device)
        disc_model_checkpoint = torch.load(config['checkpoint']['disc_checkpoint_path'], map_location=device)
        model.load_state_dict(model_checkpoint['model_state_dict'])
        disc_model.load_state_dict(disc_model_checkpoint['model_state_dict'])
        resume_epoch = model_checkpoint['epoch']
        if resume_epoch >= config['common']['max_epoch']:
            raise ValueError(f"resume epoch {resume_epoch} is larger than total epochs {config['common']['max_epoch']}")
        #logger.info(f"load chenckpoint of model and disc_model, resume from {resume_epoch}")

    if config['checkpoint']['resume'] and 'scheduler_state_dict' in model_checkpoint.keys() and 'scheduler_state_dict' in disc_model_checkpoint.keys(): 
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        optimizer_disc.load_state_dict(disc_model_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
        disc_scheduler.load_state_dict(disc_model_checkpoint['scheduler_state_dict'])
        logger.info(f"load optimizer and disc_optimizer state_dict from {resume_epoch}")

    writer = SummaryWriter(log_dir=f"{config['checkpoint']['save_folder']}/runs")  
    logger.info(f'Saving tensorboard logs to {Path(writer.log_dir).resolve()}')
   
    start_epoch = max(1,resume_epoch+1) # start epoch is 1 if not resume
    # instantiate loss balancer
    #test(0, model, disc_model, testloader, config, writer)
    # 初始化 EMA 平滑损失与 λ 权重

    for epoch in range(start_epoch, config['common']['max_epoch']+1):
        train_one_step(config,epoch, optimizer,optimizer_disc, model,disc_model,trainloader,writer,scheduler,disc_scheduler,balancer)
        if epoch % config['common']['test_interval'] == 0:
            test(epoch, model, trainloader,testloader, config, writer)
        # save checkpoint and epoch
        if epoch % config['common']['save_interval'] == 0:
            save_master_checkpoint(epoch, model, optimizer, scheduler, f"{config['checkpoint']['save_folder']}/bs{config['datasets']['batch_size']}_epoch{epoch}_lr{config['optimization']['lr']}.pt")
            save_master_checkpoint(epoch, disc_model, optimizer_disc, disc_scheduler, f"{config['checkpoint']['save_folder']}/bs{config['datasets']['batch_size']}_epoch{epoch}_disc_lr{config['optimization']['lr']}.pt") 
def main(config): 
    train(config)  # set single gpu train 


if __name__ == '__main__':
    with open("/home/xinyue/code/TCEmodel/stage3/config/config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    main(config)