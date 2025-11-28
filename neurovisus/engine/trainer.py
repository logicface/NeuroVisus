import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import accuracy_score
import numpy    as np

from ..models import get_model
from ..utils.logger import setup_logger
from .hooks import EarlyStopping
from ..models.losses import CompositeLoss
from ..models.legacy import BaseFusionModel

BaseFusionModel.calculate_odi

# 兼容 AMP
try:
    from torch.amp import autocast, GradScaler
    device_type = 'cuda'
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    device_type = 'cuda'

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class Trainer:
    def __init__(self, config, dataset):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        # 1. Logger
        log_dir = os.path.join(self.cfg['output_dir'], 'logs')
        self.logger, _ = setup_logger(log_dir, self.cfg['exp_name'])
        self.logger.info(f"🚀 初始化 Trainer: {self.cfg['exp_name']}")

        # 2. Data
        train_size = int(len(dataset) * (1 - self.cfg['val_split']))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size], 
                                          generator=torch.Generator().manual_seed(42))
        
        self.train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'], shuffle=True,
                                       num_workers=self.cfg['num_workers'], pin_memory=self.cfg['pin_memory'])
        self.val_loader = DataLoader(val_set, batch_size=self.cfg['batch_size'], shuffle=False,
                                     num_workers=self.cfg['num_workers'], pin_memory=self.cfg['pin_memory'])

        # 3. Model
        self.model = get_model(
            self.cfg['arch_name'], 
            use_bio=self.cfg['use_bio'],
            bio_channels=self.cfg['bio_channels']
        ).to(self.device)

        # 4. Optimization
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if 'vis_encoder.vit' in name or 'bio_encoder.vit' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.cfg['lr'] * 0.1},
            {'params': head_params, 'lr': self.cfg['lr']}
        ], weight_decay=self.cfg['weight_decay'])
        
        self.scaler = GradScaler()
        # self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion = CompositeLoss(ordinal_weight=1.0, smoothing=0.1)
        self.criterion_mse = nn.MSELoss()

        # 5. Callbacks
        self.best_model_path = os.path.join(self.cfg['output_dir'], f"{self.cfg['exp_name']}_best.pth")
        self.early_stopping = EarlyStopping(
            patience=self.cfg['patience'], path=self.best_model_path, trace_func=self.logger.info
        )

    def _add_bio_noise(self, bio):
        if bio is None: return None
        noise = torch.randn_like(bio) * self.cfg['bio_noise_std']
        return bio + noise

    def train_epoch(self, epoch):
        # 冻结策略
        if hasattr(self.model, 'freeze_visual_backbone'):
            if epoch < self.cfg['warmup_epochs']:
                if epoch == 0: 
                    self.model.freeze_visual_backbone()
                    self.logger.info("❄️ Backbone Frozen")
            elif epoch == self.cfg['warmup_epochs']: 
                self.model.unfreeze_visual_backbone()
                self.logger.info("🔥 Backbone Unfrozen")

        self.model.train()
        total_loss = 0.0
        loop = tqdm(self.train_loader, desc=f"Ep {epoch+1}", leave=False)
        
        for batch in loop:
            videos = batch['video'].to(self.device, non_blocking=True) 
            labels = batch['label'].to(self.device, non_blocking=True)
            bios = batch['bio'].to(self.device, non_blocking=True)
            
            # Bio Augmentation
            current_bio = bios
            if self.cfg['use_bio']: 
                current_bio = self._add_bio_noise(current_bio)
                if random.random() < self.cfg['bio_dropout_prob']: current_bio = None 

            self.optimizer.zero_grad()
            with autocast(device_type=device_type, enabled=True):
                logits = self.model(videos, bio=current_bio)
                # loss_ce = self.criterion_ce(logits, labels)
                loss = self.criterion(logits, labels)
                pred_odi = self.model.calculate_odi(logits)
                true_odi = labels.float() / 4.0
                loss_mse = self.criterion_mse(pred_odi, true_odi)
                loss = loss + self.cfg['mse_weight'] * loss_mse

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        all_preds_odi, all_true_odi = [], []
        all_preds_cls, all_true_cls = [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                videos = batch['video'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                bios = batch['bio'].to(self.device, non_blocking=True)
                
                # 50% 概率触发 Mixup
                if np.random.random() < 0.5:
                    mixed_videos, labels_a, labels_b, lam = mixup_data(videos, labels, alpha=1.0)
                    
                    # 如果有 Bio 信号，也可以 Mixup，或者保持原样(看你选择)
                    # 这里简单起见，我们只 Mixup 视频，Bio 用原图的(或者也mix)
                    # mixed_bios = lam * bios + (1 - lam) * bios[index] # 需要获取index
                    
                    with autocast(device_type=device_type, enabled=True):
                        # 前向传播
                        logits = self.model(mixed_videos, bio=bios) # 暂时不 mix bio
                        
                        # Mixup Loss 计算
                        loss_a = self.criterion(logits, labels_a) + self.cfg['mse_weight'] * self.criterion_mse(self.model.calculate_odi(logits), labels_a.float()/4.0)
                        loss_b = self.criterion(logits, labels_b) + self.cfg['mse_weight'] * self.criterion_mse(self.model.calculate_odi(logits), labels_b.float()/4.0)
                        loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    # 原来的正常训练逻辑
                    with autocast(device_type=device_type, enabled=True):
                        logits = self.model(videos, bio=bios)
                        loss_ce = self.criterion(logits, labels)
                        pred_odi = self.model.calculate_odi(logits)
                        true_odi = labels.float() / 4.0
                        loss_mse = self.criterion_mse(pred_odi, true_odi)
                        loss = loss_ce + self.cfg['mse_weight'] * loss_mse
                
                with autocast(device_type=device_type, enabled=True):
                    logits = self.model(videos, bio=bios if self.cfg['use_bio'] else None)
                    pred_odi = self.model.calculate_odi(logits)
                    true_odi = labels.float() / 4.0
                    loss = self.criterion(logits, labels) + self.cfg['mse_weight'] * self.criterion_mse(pred_odi, true_odi)
                    val_loss += loss.item()
                    
                    all_preds_odi.extend(pred_odi.cpu().numpy())
                    all_true_odi.extend(true_odi.cpu().numpy())
                    _, preds_cls = torch.max(logits, 1)
                    all_preds_cls.extend(preds_cls.cpu().numpy())
                    all_true_cls.extend(labels.cpu().numpy())
        
        metrics = {
            'loss': val_loss / len(self.val_loader),
            'mae': np.mean(np.abs(np.array(all_preds_odi) - np.array(all_true_odi))),
            'pcc': np.corrcoef(all_preds_odi, all_true_odi)[0, 1] if len(all_preds_odi)>1 else 0,
            'acc': accuracy_score(all_true_cls, all_preds_cls)
        }
        return metrics

    def run(self):
        for epoch in range(self.cfg['epochs']):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            self.logger.info(
                f"Epoch {epoch+1:02d} | TrainLoss: {train_loss:.3f} | "
                f"ValLoss: {val_metrics['loss']:.3f} | MAE: {val_metrics['mae']:.4f} | "
                f"Acc: {val_metrics['acc']:.2%}"
            )
            
            self.early_stopping(val_metrics['loss'], self.model)
            if self.early_stopping.early_stop:
                self.logger.info("🛑 Early Stopping Triggered")
                break
        
