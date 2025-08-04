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
import itertools
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from torch.utils.data import RandomSampler,SequentialSampler
import customAudioDataset as data
from customAudioDataset import collate_fn

from encodec_model import Encodec_model
from msstftd import MultiScaleSTFTDiscriminator
from scheduler import WarmupCosineLrScheduler
from utils import (count_parameters, save_master_checkpoint, set_seed)
from losses import disc_loss, total_loss
from balancer import Balancer
import numpy as np
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def train_one_step(epoch, optimizer,optimizer_disc, model, disc_model, trainloader,config,
            scheduler,disc_scheduler,writer,balancer):
    """train one step function
    Args:
        epoch (int): current epoch
        optimizer (_type_) : generator optimizer
        optimizer_disc (_type_): discriminator optimizer
        model (_type_): generator model
        disc_model (_type_): discriminator model
        trainloader (_type_): train dataloader
        config (_type_): hydra config file
        scheduler (_type_): adjust generate model learning rate
        disc_scheduler (_type_): adjust discriminator model learning rate
        warmup_scheduler (_type_): warmup learning rate
    """
    model.train()
    disc_model.train()
    data_length=len(trainloader)
 
    # Initialize variables to accumulate losses  

    for idx,wave in enumerate(trainloader):
        # warmup learning rate, warmup_epoch is defined in config file,default is 5
        wave = wave.to(device)
        optimizer.zero_grad()

        output =  model(wave) #output: [B, 1, T]: eg. [2, 1, 203760] | loss_w: [1] 
        logits_real, fmap_real = disc_model(wave)
        logits_fake, fmap_fake = disc_model(output)
        
        losses_g = total_loss(
            fmap_real, 
            logits_fake, 
            fmap_fake, 
            wave, 
            output, 
            sample_rate=config['model']['sample_rate'],
        ) 

        if balancer is not None:
            balancer.backward(losses_g, output, retain_graph=True)
            # naive loss summation for metrics below
            loss_g = sum([l * balancer.weights[k] for k, l in losses_g.items()])
        else:
            # without balancer: loss = 3*l_g + 3*l_feat + (l_t / 10) + l_f
            # loss_g = torch.tensor([0.0], device='cuda', requires_grad=True)
            loss_g = 3*losses_g['l_g'] + 3*losses_g['l_feat'] + losses_g['l_t']/10 + losses_g['l_f'] 
        loss_g.backward()
        optimizer.step()

        # only update discriminator with probability from paper (configure)
        optimizer_disc.zero_grad()
        train_discriminator = torch.BoolTensor([config['model']['train_discriminator']
                               and epoch >= config['lr_scheduler']['warmup_epoch'] 
                               and random.random() < float(eval(str(config['model']['train_discriminator'])))]).to(device)

        if train_discriminator:
            logits_real, _ = disc_model(wave)
            logits_fake, _ = disc_model(output.detach()) # detach to avoid backpropagation to model
            loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss
            loss_disc.backward() 
            optimizer_disc.step()

        scheduler.step()
        disc_scheduler.step()

        log_msg = (  
            f"Epoch {epoch} {idx+1}/{data_length}\tAvg loss_G: {loss_g.item():.4f} /{(idx + 1)}\tlr_G: {optimizer.param_groups[0]['lr']:.6e}\tlr_D: {optimizer_disc.param_groups[0]['lr']:.6e}\t"  
        ) 
        writer.add_scalar('Train/Loss_g', loss_g, (epoch-1) * len(trainloader) + idx) 
        writer.add_scalar('Train/l_g', losses_g['l_g'], (epoch-1) * len(trainloader) + idx)  
        writer.add_scalar('Train/l_feat', losses_g['l_feat'], (epoch-1) * len(trainloader) + idx)  
        writer.add_scalar('Train/l_f', losses_g['l_f'], (epoch-1) * len(trainloader) + idx)   
        writer.add_scalar('Train/l_t', losses_g['l_t'], (epoch-1) * len(trainloader) + idx)
        if train_discriminator:
            log_msg += f"loss_disc: {loss_disc.item() / (idx + 1) :.4f}"  
            writer.add_scalar('Train/Loss_Disc', loss_disc.item(), (epoch-1) * len(trainloader) + idx) 
        logger.info(log_msg) 

@torch.no_grad()
def test(epoch, model,disc_model, testloader, config, writer):
    model.eval()
    disc_model.eval()
    # loss_g = torch.tensor([0.0], device=device, requires_grad=True)
    # for idx, input in enumerate(testloader):
    #     input = input.to(device)
    #     output = model(input)
    #     logits_real, fmap_real = disc_model(input)
    #     logits_fake, fmap_fake = disc_model(output)
    #     loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss
    # log_msg = (f'| TEST | epoch: {epoch} | loss_g: {loss_g.item():.4f}| loss_disc: {loss_disc.item():.4f}') 
    # writer.add_scalar(f'Test/Loss_g', loss_g.item(), epoch)  
    # writer.add_scalar('Test/Loss_Disc', loss_disc.item(), epoch)
    # logger.info(log_msg)

    # # save a sample reconstruction (not cropped)
    input_wav = testloader.dataset.get()[1].unsqueeze(0).to(device)

    output= model(input_wav)
    # summarywriter can't log stereo files 😅 so just save examples
    sp = Path(config['checkpoint']['save_folder'])
    torchaudio.save(sp/f'GT.wav', input_wav.squeeze(0).cpu(), config['model']['sample_rate'])
    torchaudio.save(sp/f'Reconstruction.wav', output.squeeze(0).cpu(), config['model']['sample_rate'])

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

    model = Encodec_model(
        config['model']['target_bandwidths'],
        config['model']['sample_rate'], 
        config['model']['channels'],
        causal=config['model']['causal'], 
        model_norm=config['model']['norm'],
        dimension=int(config['model']['dimension']), 
        ratios=config['model']['ratios'],
    )
    disc_model = MultiScaleSTFTDiscriminator(
        in_channels=config['model']['channels'],
        out_channels=config['model']['channels'],
        filters=config['model']['filters'],
        hop_lengths=config['model']['disc_hop_lengths'],
        win_lengths=config['model']['disc_win_lengths'],
        n_ffts=config['model']['disc_n_ffts'],
    )
    # log model, disc model parameters and train mode
    # logger.info(model)
    # logger.info(disc_model)
    # logger.info(config)
    # logger.info(f"Decoupling Model Parameters: {count_parameters(model)} | Disc Model Parameters: {count_parameters(disc_model)}")
    # logger.info(f"model train mode :{model.training} | disc_model train mode :{disc_model.training}")

    
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
    
    
    params = [p for p in model.parameters() if p.requires_grad]
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    optimizer = optim.Adam([{'params': params, 'lr': config['optimization']['lr']}], betas=(0.5, 0.9))
    optimizer_disc = optim.Adam([{'params':disc_params, 'lr': config['optimization']['disc_lr']}], betas=(0.5, 0.9))
    #退火策略
    scheduler = WarmupCosineLrScheduler(optimizer, max_iter=config['common']['max_epoch']*len(trainloader), eta_ratio=0.1, warmup_iter=config['lr_scheduler']['warmup_epoch']*len(trainloader), warmup_ratio=1e-4)
    disc_scheduler = WarmupCosineLrScheduler(optimizer_disc, max_iter=config['common']['max_epoch']*len(trainloader), eta_ratio=0.1, warmup_iter=config['lr_scheduler']['warmup_epoch']*len(trainloader), warmup_ratio=1e-4)
    

    resume_epoch = 0
    if config['checkpoint']['resume']:
        # check the checkpoint_path
        assert config['checkpoint']['checkpoint_path'] != '', "resume path is empty"
        assert config['checkpoint']['disc_checkpoint_path'] != '', "disc resume path is empty"

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
    balancer = Balancer(dict(config['balancer']['weights'])) if hasattr(config, 'balancer') else None
    if balancer:
        logger.info(f'Loss balancer with weights {balancer.weights} instantiated')
    #test(0, model, disc_model, testloader, config, writer)
    for epoch in range(start_epoch, config['common']['max_epoch']+1):
        train_one_step(
            epoch, optimizer,optimizer_disc, model, disc_model, trainloader,config,
            scheduler,disc_scheduler,writer,balancer)
        if epoch % config['common']['test_interval'] == 0:
            test(epoch,model,disc_model,testloader,config,writer)
        # save checkpoint and epoch
        if epoch % config['common']['save_interval'] == 0:
            save_master_checkpoint(epoch, model, optimizer, scheduler, f"{config['checkpoint']['save_folder']}/bs{config['datasets']['batch_size']}_epoch{epoch}_lr{config['optimization']['lr']}.pt")  
            save_master_checkpoint(epoch, disc_model, optimizer_disc, disc_scheduler, f"{config['checkpoint']['save_folder']}/bs{config['datasets']['batch_size']}_epoch{epoch}_disc_lr{config['optimization']['lr']}.pt") 

def main(config): 
    train(config)  # set single gpu train 


if __name__ == '__main__':
    with open("/home/xinyue/code/TCEmodel/stage1/config/config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    main(config)