"""Trainer for RhythmMamba."""
import os
import numpy as np
import torch
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import neural_methods.trainer.utils as utils

from tqdm import tqdm
from evaluation.post_process import calculate_hr
from evaluation.metrics import calculate_metrics
from neural_methods.model.RhythmMamba import  RhythmMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.TorchLossComputer import Hybrid_Loss, Consistency_Loss, Freq_Consistency_Loss

class RhythmMambaTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = 0
        self.data_dict = {}
        self.dataset = config.TRAIN.DATA.DATASET
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1
        if config.TOOLBOX_MODE == "train_and_test":
            self.model = RhythmMamba().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.num_train_batches = len(data_loader["train"])
            self.criterion = Hybrid_Loss()
            # self.consistency_criterion = Consistency_Loss()
            self.consistency_criterion = Freq_Consistency_Loss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = RhythmMamba().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("EfficientPhys trainer initialized in incorrect toolbox mode!")
            
    
    def _set_bn_track_stats(self, track: bool):
        """Helper to freeze/unfreeze BatchNorm running stats during augmented forward pass."""
        for m in self.model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.track_running_stats = track
    
    
    # ---- Loss Time ---
    # def train(self, data_loader):
    #     """Training routine for model (Version 2: Split Forward BN-Safe + Dynamic Clean Fine-tuning)"""
    #     if data_loader["train"] is None:
    #         raise ValueError("No data for train")

    #     total_iter = self.max_epoch_num * self.num_train_batches

    #     for epoch in range(self.max_epoch_num):
    #         print(f"\n====Training Epoch: {epoch}====")
    #         self.model.train()

    #         running_total = 0.0
    #         running_main = 0.0
    #         running_cons = 0.0

    #         tbar = tqdm(data_loader["train"], dynamic_ncols=True)
    #         for idx, batch in enumerate(tbar):
    #             tbar.set_description(f"Train epoch {epoch}")

    #             data = batch[0].float().to(self.device)
    #             labels = batch[1].float().to(self.device)
    #             N = data.shape[0]

    #             self.optimizer.zero_grad()

    #             # 1. Forward pass on clean original data (updates BatchNorm running stats normally)
    #             pred_orig, feat_orig = self.model(data)
    #             pred_orig = (pred_orig - torch.mean(pred_orig, dim=-1, keepdim=True)) / (
    #                 torch.std(pred_orig, dim=-1, keepdim=True) + 1e-8
    #             )

    #             loss_orig = 0.0
    #             for ib in range(N):
    #                 loss_orig = loss_orig + self.criterion(
    #                     pred_orig[ib], labels[ib], epoch, self.config.TRAIN.DATA.FS, self.diff_flag
    #                 )
    #             loss_orig = loss_orig / N

    #             if self.config.TRAIN.AUG:
    #                 data_aug, labels_aug = self.data_augmentation_ver2(data, labels)

    #                 # 2. Forward pass on augmented data WITHOUT polluting BatchNorm running stats
    #                 self._set_bn_track_stats(False)
    #                 pred_aug, feat_aug = self.model(data_aug)
    #                 self._set_bn_track_stats(True)

    #                 pred_aug = (pred_aug - torch.mean(pred_aug, dim=-1, keepdim=True)) / (
    #                     torch.std(pred_aug, dim=-1, keepdim=True) + 1e-8
    #                 )

    #                 loss_aug = 0.0
    #                 for ib in range(N):
    #                     loss_aug = loss_aug + self.criterion(
    #                         pred_aug[ib], labels_aug[ib], epoch, self.config.TRAIN.DATA.FS, self.diff_flag
    #                     )
    #                 loss_aug = loss_aug / N

    #                 # 3. Frequency Consistency Loss
    #                 consistency_loss = self.consistency_criterion(feat_orig, feat_aug)

            
    #                 # 4. Bell-shaped lambda_factor synchronized with OneCycleLR (pct_start = 0.3)
    #                 current_iter = epoch * self.num_train_batches + idx
    #                 r = current_iter / max(total_iter - 1, 1)
    #                 if r <= 0.3:
    #                     lambda_factor = 0.5 * (1.0 - np.cos(np.pi * (r / 0.3)))
    #                 else:
    #                     lambda_factor = 0.5 * (1.0 + np.cos(np.pi * ((r - 0.3) / 0.7)))

    #                 # Smoothly shift focus toward clean signal in late epochs to sharpen PSD peak (boost SNR)
    #                 main_loss = (loss_orig + lambda_factor * loss_aug) / (1.0 + lambda_factor)
    #                 loss = main_loss + 0.1 * lambda_factor * consistency_loss
    #             else:
    #                 main_loss = loss_orig
    #                 loss = main_loss
    #                 consistency_loss = torch.tensor(0.0, device=self.device)

    #             loss.backward()
    #             self.optimizer.step()
    #             self.scheduler.step()

    #             running_total += loss.item()
    #             running_main += main_loss.item()
    #             running_cons += consistency_loss.item()

    #             tbar.set_postfix({
    #                 "total": f"{running_total / (idx + 1):.4f}",
    #                 "main": f"{running_main / (idx + 1):.4f}",
    #                 "cons": f"{running_cons / (idx + 1):.4f}",
    #             })

    #         self.save_model(epoch)
    #         if not self.config.TEST.USE_LAST_EPOCH:
    #             valid_loss = self.valid(data_loader)
    #             print("validation loss: ", valid_loss)
    #             if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
    #                 self.min_valid_loss = valid_loss
    #                 self.best_epoch = epoch
    #                 print(f"Update best model! Best epoch: {self.best_epoch}")

    #     if not self.config.TEST.USE_LAST_EPOCH:
    #         print(f"best trained epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")


    # ---- Loss Freq ---
    def train(self, data_loader):
        """Training routine for model (Version 2: Split Forward BN-Safe + Dynamic Clean Fine-tuning)"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        total_iter = self.max_epoch_num * self.num_train_batches

        for epoch in range(self.max_epoch_num):
            print(f"\n====Training Epoch: {epoch}====")
            self.model.train()

            epoch_representation_orig = []
            epoch_representation_aug = []
            epoch_freqs = None

            running_total = 0.0
            running_main = 0.0
            running_cons = 0.0

            tbar = tqdm(data_loader["train"], dynamic_ncols=True)
            for idx, batch in enumerate(tbar):
                tbar.set_description(f"Train epoch {epoch}")

                data = batch[0].float().to(self.device)
                labels = batch[1].float().to(self.device)
                N = data.shape[0]

                self.optimizer.zero_grad()
                
                if self.config.TRAIN.AUG:
                    data, labels = self.temporal_hr_resample(data, labels, batch[2], batch[3])
                    data = (data - data.mean(dim=(1, 2, 3, 4), keepdim=True)) / (data.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-6)
                    labels = (labels - labels.mean(dim=-1, keepdim=True)) / (labels.std(dim=-1, keepdim=True) + 1e-6)

                # 1. Forward pass on clean original data (updates BatchNorm running stats normally)
                pred_orig, feat_orig = self.model(data)
                pred_orig = (pred_orig - torch.mean(pred_orig, dim=-1, keepdim=True)) / (
                    torch.std(pred_orig, dim=-1, keepdim=True) + 1e-8
                )

                loss_orig = 0.0
                for ib in range(N):
                    loss_orig = loss_orig + self.criterion(
                        pred_orig[ib], labels[ib], epoch, self.config.TRAIN.DATA.FS, self.diff_flag
                    )
                loss_orig = loss_orig / N

                if self.config.TRAIN.AUG:
                    data_aug, labels_aug = self.data_augmentation_ver2(data, labels)

                    # 2. Forward pass on augmented data WITHOUT polluting BatchNorm running stats
                    self._set_bn_track_stats(False)
                    pred_aug, feat_aug = self.model(data_aug)
                    self._set_bn_track_stats(True)

                    pred_aug = (pred_aug - torch.mean(pred_aug, dim=-1, keepdim=True)) / (
                        torch.std(pred_aug, dim=-1, keepdim=True) + 1e-8
                    )

                    loss_aug = 0.0
                    for ib in range(N):
                        loss_aug = loss_aug + self.criterion(
                            pred_aug[ib], labels_aug[ib], epoch, self.config.TRAIN.DATA.FS, self.diff_flag
                        )
                    loss_aug = loss_aug / N

                    # 3. Frequency Consistency Loss
                    consistency_loss = self.consistency_criterion(feat_orig, feat_aug)

                    # Store representations for PSD visualization
                    freqs, representation_orig, representation_aug = (
                        self.consistency_criterion.get_representations()
                    )
                    epoch_freqs = freqs
                    epoch_representation_orig.append(representation_orig.detach().mean(dim=0).cpu())
                    epoch_representation_aug.append(representation_aug.detach().mean(dim=0).cpu())

                    # 4. Bell-shaped lambda_factor synchronized with OneCycleLR (pct_start = 0.3)
                    current_iter = epoch * self.num_train_batches + idx
                    
                    r = current_iter / max(total_iter - 1, 1)
                    if r <= 0.3:
                        # Warmup từ 0.0 lên 1.0 trong 30% chặng đầu
                        lambda_factor = 0.5 * (1.0 - np.cos(np.pi * (r / 0.3)))
                    else:
                        # Giữ tối thiểu ở mức 0.5 thay vì tụt về 0.0 để bảo toàn tính bất biến miền cho Cross-test
                        lambda_factor = 0.5 + 0.5 * (0.5 * (1.0 + np.cos(np.pi * ((r - 0.3) / 0.7))))

                    # Smoothly shift focus toward clean signal in late epochs to sharpen PSD peak (boost SNR)
                    main_loss = (loss_orig + lambda_factor * loss_aug) / (1.0 + lambda_factor)
                    loss = main_loss + 0.1 * lambda_factor * consistency_loss
                else:
                    main_loss = loss_orig
                    loss = main_loss
                    consistency_loss = torch.tensor(0.0, device=self.device)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                running_total += loss.item()
                running_main += main_loss.item()
                running_cons += consistency_loss.item()

                tbar.set_postfix({
                    "total": f"{running_total / (idx + 1):.4f}",
                    "main": f"{running_main / (idx + 1):.4f}",
                    "cons": f"{running_cons / (idx + 1):.4f}",
                })

            # Save PSD consistency figure after every training epoch
            if epoch_freqs is not None:
                self.plot_psd_consistency(
                    epoch,
                    epoch_freqs,
                    epoch_representation_orig,
                    epoch_representation_aug,
                )

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                print("validation loss: ", valid_loss)
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print(f"Update best model! Best epoch: {self.best_epoch}")

        if not self.config.TEST.USE_LAST_EPOCH:
            print(f"best trained epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")
            

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                pred_ppg_valid, _ = self.model(data_valid)
                pred_ppg_valid = (pred_ppg_valid-torch.mean(pred_ppg_valid, axis=-1).view(-1, 1))/torch.std(pred_ppg_valid, axis=-1).view(-1, 1)    # normalize

                for ib in range(N):
                    loss = self.criterion(pred_ppg_valid[ib], labels_valid[ib], self.config.TRAIN.EPOCHS , self.config.VALID.DATA.FS , self.diff_flag)
                    valid_loss.append(loss.item())
                    valid_step += 1
                    vbar.set_postfix(loss=loss.item())
        return np.mean(np.asarray(valid_loss))


    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            predictions = dict()
            labels = dict()
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                chunk_len = self.chunk_len
                data_test, labels_test = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test, _ = self.model(data_test)
                pred_ppg_test = (pred_ppg_test-torch.mean(pred_ppg_test, axis=-1).view(-1, 1))/torch.std(pred_ppg_test, axis=-1).view(-1, 1)    # normalize
                labels_test = labels_test.view(-1, 1)
                pred_ppg_test = pred_ppg_test.view( -1 , 1)
                for ib in range(batch_size):
                    subj_index = test_batch[2][ib]
                    sort_index = int(test_batch[3][ib])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[ib * chunk_len:(ib + 1) * chunk_len]
                    labels[subj_index][sort_index] = labels_test[ib * chunk_len:(ib + 1) * chunk_len]
            print(' ')
            calculate_metrics(predictions, labels, self.config)


    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)


    def data_augmentation(self,data,labels,index1,index2):
        N, D, C, H, W = data.shape
        data_aug = np.zeros((N, D, C, H, W))
        labels_aug = np.zeros((N, D))
        rand1_vals = np.random.random(N)
        rand2_vals = np.random.random(N)
        for idx in range(N):
            index = index1[idx] + index2[idx]
            rand1 = rand1_vals[idx]
            rand2 = rand2_vals[idx]
            if rand1 < 0.5 :
                if index in self.data_dict:
                    gt_hr_fft = self.data_dict[index]
                else:
                    gt_hr_fft, _  = calculate_hr(labels[idx], labels[idx] , diff_flag = self.diff_flag , fs=self.config.VALID.DATA.FS)
                    self.data_dict[index] = gt_hr_fft
                    
                if gt_hr_fft > 90: 
                    rand3 = random.randint(0, D//2-1)
                    even_indices = torch.arange(0, D, 2)
                    odd_indices = even_indices + 1
                    data_aug[:, even_indices, :, :, :] = data[:, rand3 + even_indices// 2, :, :, :]
                    labels_aug[:, even_indices] = labels[:, rand3 + even_indices // 2]
                    data_aug[:, odd_indices, :, :, :] = (data[:, rand3 + odd_indices // 2, :, :, :] + data[:, rand3 + (odd_indices // 2) + 1, :, :, :]) / 2
                    labels_aug[:, odd_indices] = (labels[:, rand3 + odd_indices // 2] + labels[:, rand3 + (odd_indices // 2) + 1]) / 2
                elif gt_hr_fft < 75 :
                    data_aug[:, :D//2, :, :, :] = data[:, ::2, :, :, :]
                    labels_aug[:, :D//2] = labels[:, ::2]
                    data_aug[:, D//2:, :, :, :] = data_aug[:, :D//2, :, :, :]
                    labels_aug[:, D//2:] = labels_aug[:, :D//2]
                else :
                    data_aug[idx] = data[idx]
                    labels_aug[idx] = labels[idx]                                      
            else :
                data_aug[idx] = data[idx]
                labels_aug[idx] = labels[idx]
        data_aug = torch.tensor(data_aug).float()
        labels_aug = torch.tensor(labels_aug).float()
        if rand2 < 0.5:
            data_aug = torch.flip(data_aug, dims=[4])
        return data_aug, labels_aug
    
    
    def temporal_hr_resample(self, data, labels, index1, index2):
        """
        Resample the temporal dimension of the data based on the heart rate.
        """
        N, D, C, H, W = data.shape
        device = data.device
        
        data_out = data.clone()
        labels_out = labels.clone()
        
        even_indices = torch.arange(0, D, 2, device=device)
        odd_indices = even_indices + 1
        rand_vals = torch.rand(N, device=device)

        for idx in range(N):
            if rand_vals[idx] < 0.5:
                index = index1[idx] + index2[idx]
                if index in self.data_dict:
                    gt_hr_fft = self.data_dict[index]
                else:
                    lbl_np = labels[idx].detach().cpu().numpy()
                    gt_hr_fft, _ = calculate_hr(
                        lbl_np, lbl_np, 
                        diff_flag=self.diff_flag, 
                        fs=self.config.VALID.DATA.FS
                    )
                    self.data_dict[index] = gt_hr_fft

                if gt_hr_fft > 90:
                    rand3 = random.randint(0, D // 2 - 1)
                    data_out[idx, even_indices] = data[idx, rand3 + even_indices // 2]
                    labels_out[idx, even_indices] = labels[idx, rand3 + even_indices // 2]
                    
                    data_out[idx, odd_indices] = 0.5 * (
                        data[idx, rand3 + odd_indices // 2] + 
                        data[idx, rand3 + (odd_indices // 2) + 1]
                    )
                    labels_out[idx, odd_indices] = 0.5 * (
                        labels[idx, rand3 + odd_indices // 2] + 
                        labels[idx, rand3 + (odd_indices // 2) + 1]
                    )
                elif gt_hr_fft < 75:
                    data_out[idx, :D//2] = data[idx, ::2]
                    labels_out[idx, :D//2] = labels[idx, ::2]
                    data_out[idx, D//2:] = data_out[idx, :D//2]
                    labels_out[idx, D//2:] = labels_out[idx, :D//2]

        return data_out, labels_out
    
    
    def data_augmentation_ver2(self, data, labels):
        # data shape: (N, D, C, H, W), labels shape: (N, D)
        N, D = data.shape[0], data.shape[1]
        
        p_motion = self.config.TRAIN.AUG_RATE.MOTION
        p_framerate = self.config.TRAIN.AUG_RATE.FRAMERATE
        p_gamma = self.config.TRAIN.AUG_RATE.GAMMA
        p_light = self.config.TRAIN.AUG_RATE.LIGHT
        p_facial = self.config.TRAIN.AUG_RATE.FACIAL
        
        total_p = p_motion + p_framerate + p_gamma + p_light + p_facial
        assert total_p <= 1.0, f"The sum of augmentation probabilities ({total_p}) must not exceed 1.0"
        
        data_aug = data.clone()
        labels_aug = labels.clone()
        
        r = torch.rand(N, device=data.device)

        t1 = p_motion
        t2 = t1 + p_framerate
        t3 = t2 + p_gamma
        t4 = t3 + p_light
        t5 = t4 + p_facial
        
        mask_motion     = (r < t1)
        mask_frame      = (r >= t1) & (r < t2)
        mask_gamma      = (r >= t2) & (r < t3)
        mask_light      = (r >= t3) & (r < t4)
        mask_facial     = (r >= t4) & (r < t5)
        # The range of [t5, 1.0) automatically corresponds to samples that remain unchanged (Identity)
        
        if mask_motion.any() and p_motion > 0:
            data_aug[mask_motion] = utils.jitter_affine(data_aug[mask_motion])
            
        if mask_frame.any() and p_framerate > 0:
            data_aug[mask_frame] = utils.frame_drop(data_aug[mask_frame])

        if mask_gamma.any() and p_gamma > 0:
            data_aug[mask_gamma] = utils.gamma_correction(data_aug[mask_gamma])
            
        if mask_light.any() and p_light > 0:
            data_aug[mask_light] = utils.light_skin_tone_adjustment(data_aug[mask_light])
        
        if mask_facial.any() and p_facial > 0:
            data_aug[mask_facial] = utils.facial_expression_aug(data_aug[mask_facial])
            
        return data_aug, labels_aug
    
    
    def plot_psd_consistency(
        self,
        epoch,
        frequencies,
        representations_orig,
        representations_aug,
    ):
        """Plot and save the epoch-averaged original and augmented PSD."""
        output_dir = os.path.join("figures", "psd_consistency")
        os.makedirs(output_dir, exist_ok=True)

        frequencies = frequencies.detach().cpu().numpy()
        representation_orig = torch.stack(representations_orig).mean(dim=0).numpy()
        representation_aug = torch.stack(representations_aug).mean(dim=0).numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(frequencies, representation_orig, label="Original")
        plt.plot(frequencies, representation_aug, label="Augmented")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power spectral density")
        plt.title(f"PSD Consistency - Epoch {epoch}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(
            output_dir,
            f"{self.model_file_name}_epoch_{epoch}.png",
        )
        plt.savefig(output_path, dpi=200)
        plt.close()
        print("Saved PSD consistency plot:", output_path)