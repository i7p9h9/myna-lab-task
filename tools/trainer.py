import os
import torch
import numpy as np

from tools.models.ResNetSE34Q import MainModel
from tools.losses.ctc import LossFunction

from tools.generic.encoder import NumEncoder
from tools.generic.timer import Timer
from tools.generic.misc import greedy_decoder
from tools.generic.metrics import cer


class Trainer(torch.nn.Module):
    def __init__(self, 
                 save_folder,
                 epoch_steps=1000,
                 validation_period=1000,
                 temperature=0.5,
                 threshold=0.95):
        super(Trainer, self).__init__()
        self.temperature = temperature
        self.threshold = threshold
        
        self.save_folder = save_folder
        self.log_file = None
        self.epoch_steps = epoch_steps
        self.validation_period = validation_period
        
        self.model = MainModel(nOut=32).cuda()
        self.loss = LossFunction(nOut=32, nClasses=20, blank=0).cuda()
        self.unsupervised_criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001, weight_decay = 0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1, 
            gamma=0.2,
            verbose=True)
        
        self.timer = Timer()
        self.encoder = NumEncoder()

        self.min_cer = np.inf
        
    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype("float32")).cuda()
            
        with torch.no_grad():
            out = self.model.eval().forward(x.cuda())
            self.model.train()
            logits = torch.tensordot(out, self.loss.W, dims=1)
            
        return logits
        
    def logme(self, n_epoch, message=None):
        if self.save_folder is None:
            return
        save_folder = os.path.join(self.save_folder, "epoch-{}".format(n_epoch))
        os.makedirs(save_folder, exist_ok=True)
        
        if message is not None:
            log_file = os.path.join(save_folder, "log.txt")
            with open(log_file, "a+") as f:
                f.write(message + "\n")
        torch.save(self.state_dict(), os.path.join(save_folder, "model-{:02d}.torch".format(n_epoch)))
    
    def epoch_step(self, 
                   n_epoch, 
                   generator_labeled,
                   generator_unlabeled,
                   generator_valid,
                   unsupervised_interval=10):
        self.train()
        
        loss_sup_avg = 0
        loss_unsup_avg = 0
        loss_avg = 0
        index = 0
        step_size = generator_labeled._batch_size
        
        self.timer.tic()
        for n_step in range(self.epoch_steps):
            self.zero_grad()
            
            l_supervised = self.supervised_step(n_step, generator_labeled)
            
            if np.mod(n_step + 1, unsupervised_interval) == 0:
                l_unsupervised = self.unsupervised_step(n_step, generator_unlabeled)
                l = l_supervised + 1.0 * l_unsupervised
            else:
                l = l_supervised
                
            l.backward(l)
            self.optimizer.step()

            loss_sup_avg += float(l_supervised.detach().cpu())
            try:
                loss_unsup_avg += float(l_unsupervised.detach().cpu())
            except:
                pass
                
            loss_avg += float(l.detach().cpu())
            index   += step_size

            info_message = " ".join(["\repoch {:d};",
                                "Processing {:d} / {:d};",
                                "Loss {:0.4f}",
                                "Loss S {:0.4f}",
                                "Loss U {:0.4f}",
                                "- {:0.2f} Hz"]).format(
                n_epoch,
                index,
                step_size * self.epoch_steps,
                loss_avg / (n_step + 1),
                loss_sup_avg / (n_step + 1),
                loss_unsup_avg / (n_step + 1),
                step_size / self.timer.tictoc())
            
            print(info_message, end='\r', flush=True)
            if np.mod(n_step + 1, self.epoch_steps // 10) == 0:
                print()
                self.logme(n_epoch, message=info_message)
            
            if np.mod(n_step + 1, self.validation_period) == 0:
                self.validation(generator_valid)
        
        self.scheduler.step()
        self.logme(n_epoch, message=info_message)

    def supervised_step(self, n_step, generator):
        x, y, target_length = generator[n_step]
        out = self.model.forward(x.cuda())
        l = self.loss.forward(out, y.cuda(), target_length.cuda())
        
        return l

    def unsupervised_step(self, n_step, generator, repeat=2):
        with torch.no_grad():
            x, *_ = generator[n_step]
            x_es = x[:generator._batch_size]
            x_hard = x[generator._batch_size:]
            logits = self.predict(x_es)
            
            pseudo_label = torch.softmax(logits.detach() / self.temperature, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.threshold).float()

        outs_u = self.model.forward(x_hard.cuda())
        logits_u = torch.tensordot(outs_u, self.loss.W, dims=1)
        targets_u = torch.repeat_interleave(targets_u, repeat, dim=0)
        
        mask = torch.repeat_interleave(mask, repeat, dim=0)
        l = (F.cross_entropy(logits_u.transpose(1, 2), targets_u, reduction='none') * mask).mean()
        
        return l
    
    def validation(self, generator):
        print("start validation ...")
        valid_cer = []
        for n in range(len(generator)):
            x, y = generator[n]
            p = self.predict(x)
            decoded_seq = greedy_decoder(p.cpu().numpy(), blank_label=0)
            seq2num = self.encoder((np.asarray(decoded_seq).squeeze() - 1).tolist())
            
            valid_cer.append(cer(str(y), str(seq2num)))
        
        avg_cer = sum(valid_cer) / len(valid_cer)
        
        message = "Validation SER: {:0.4f}".format(avg_cer)
        print(message)
        self.logme(n_epoch, message=message)

        if avg_cer < self.min_cer:
            self.min_cer = avg_cer
            torch.save(self.state_dict(), os.path.join(self.save_folder, "final.torch"))

        return avg_cer
         
