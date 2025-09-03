import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import pandas as pd
import math
#from mha import MultiHeadAttention
from typing import Optional

import copy
import torchmetrics
from timeit import default_timer as timer

from pathlib import Path
import sys
root_dir = Path(__file__).parents[1] # Goes up 2 levels to ANOMALYDETECTIONHADES
sys.path.append(str(root_dir))

from utils import helper_functions as hf

class PaddingTon_9_Model_V1(nn.Module):
    """
    Given an input of size (batch,9,768), outputs (batch) floating point values of the normalized
    interaction point between -1 and 1.
    dim=1 represents the input channels or detectors (UFT1,MFT1,MFT2) with their axis (x,u,v)
    dim=2 is the fiber number which will be 2 for a hit, 1 for a miss, and 0 for padding.
    UFT1 is not padded, MFT1 and MFT2 are padded from 512 to 768.
    """
    def __init__(self,input_size, hidden_units=8, output_dim=1,kernel_size_convolution=2,kernel_size_maxpool=2,stride=1,padding=1):
        super().__init__()
        # self.Embed=nn.Embedding(2, embedding_dim=1,padding_idx=0)

        
        
        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=hidden_units,
                      kernel_size=kernel_size_convolution, # how big is the square that's going over the image?
                      stride=stride, # default, skips one each kernel
                      padding=padding),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),           #padding adds n-pixels to edge
            nn.Conv1d(in_channels=hidden_units,
                      out_channels=hidden_units*2,
                      kernel_size=kernel_size_convolution,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size_maxpool) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv1d(hidden_units*2, hidden_units*4, kernel_size_convolution,stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_units*4, hidden_units*6, kernel_size_convolution,stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size_maxpool)
        )
        self.block_3 = nn.Sequential(
            nn.Conv1d(hidden_units*6, hidden_units*8, kernel_size_convolution,stride=stride, padding=padding),
            #nn.BatchNorm2d(hidden_units*8),
            #nn.Tanh(),
            nn.ReLU(),
            #nn.Dropout(p=dropout_p), 
            nn.Conv1d(hidden_units*8, hidden_units*10, kernel_size_convolution,stride=stride, padding=padding),
            #nn.BatchNorm2d(hidden_units*10),
            #nn.Tanh(),
            nn.ReLU(),
            #nn.Dropout(p=dropout_p), 
            nn.MaxPool1d(kernel_size_maxpool)
        )
        self.block_4 = nn.Sequential(
            nn.Conv1d(hidden_units*10, hidden_units*12, kernel_size_convolution,stride=stride, padding=padding),
            #nn.BatchNorm2d(hidden_units*8),
            #nn.Tanh(),
            nn.ReLU(),
            #nn.Dropout(p=dropout_p), 
            nn.Conv1d(hidden_units*12, hidden_units*14, kernel_size_convolution,stride=stride, padding=padding),
            #nn.BatchNorm2d(hidden_units*10),
            #nn.Tanh(),
            nn.ReLU(),
            #nn.Dropout(p=dropout_p), 
            nn.MaxPool1d(kernel_size_maxpool)
        )
        with torch.inference_mode():
            dummy_input = torch.zeros(1,1,input_size)  # Batch size 1, input_shape channels, detectors, hits
            dummy_output = self.block_1(dummy_input)
            print(dummy_output.shape)
            dummy_output = self.block_2(dummy_output)
            print(dummy_output.shape)
            dummy_output=self.block_3(dummy_output)
            print(dummy_output.shape)
            dummy_output=self.block_4(dummy_output)
            print(dummy_output.shape)
            in_features = dummy_output.view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            #add GRU
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our input data.
            #how to calculate in_features??
            nn.Linear(in_features=in_features, #¡¡¡¡wanna make a code to do this for me!!!
                      out_features=output_dim)
            # nn.Tanh() #could muck up the output for preds around 1,-1
        )

    def forward(self, x: torch.Tensor):
        x=x.squeeze()
        # x=x.to(torch.float32).permute(0,3,1,2)
        # x=self.Embed(x.to(torch.long)).permute(0,3,1,2)
        x=x.to(torch.float32).unsqueeze(1)
        x = self.classifier(self.block_4(self.block_3(self.block_2(self.block_1(x)))))
        return x.squeeze()  # Remove the last dimension to match y shape


class HADES_5S_1D(nn.Module):
    def __init__(self, hidden_units=8, output_dim=9, kernel_size_convolution=3, 
                 kernel_size_maxpool=2, stride=1, padding=1):
        super().__init__()
        
        # Shared feature extractor for all 5 histogram types
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=hidden_units,
                     kernel_size=kernel_size_convolution, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units*2,
                     kernel_size=kernel_size_convolution, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size_maxpool),
            
            # Block 2
            nn.Conv1d(hidden_units*2, hidden_units*4, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.Conv1d(hidden_units*4, hidden_units*6, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size_maxpool),
            
            # Block 3
            nn.Conv1d(hidden_units*6, hidden_units*8, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.Conv1d(hidden_units*8, hidden_units*10, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size_maxpool),
            
            # Block 4
            nn.Conv1d(hidden_units*10, hidden_units*12, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.Conv1d(hidden_units*12, hidden_units*14, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1)  # Global average pooling
        )
        
        # Calculate the total features from all 5 histograms
        total_features = (hidden_units * 14)
        
        # Final classifier that combines information from ALL 5 histograms
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_features, total_features//2),  # Intermediate layer
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(total_features//2, output_dim)       # Final 9-class prediction
        )

    def forward(self, x):
        # Extract each histogram (assuming x is a dictionary)
        START1 = x["START_beam_mod1"].unsqueeze(1).float()  # shape: [batch, 1, 20]
        START2 = x["START_beam_mod2"].unsqueeze(1).float()  # shape: [batch, 1, 20]  
        RICH_TRENDS = x["RICH_trends"].unsqueeze(1).float() # shape: [batch, 1, 50]
        TOF_MULT = x["ToF_multiplicity"].unsqueeze(1).float() # shape: [batch, 1, 100]
        TOF_SUM = x["ToF_sum"].unsqueeze(1).float()         # shape: [batch, 1, 150]
        
        # Extract features from each histogram
        features1 = self.classifier(self.feature_extractor(START1)).unsqueeze(2)   # shape: [batch, 1]
        features2 = self.classifier(self.feature_extractor(START2)).unsqueeze(2)
        features3 = self.classifier(self.feature_extractor(RICH_TRENDS)).unsqueeze(2)
        features4 = self.classifier(self.feature_extractor(TOF_MULT)).unsqueeze(2)
        features5 = self.classifier(self.feature_extractor(TOF_SUM)).unsqueeze(2)
        
        # Concatenate all features along channel dimension
        combined_features = torch.cat([features1, features2, features3, features4, features5], dim=2) # shape: [batch,hidden_features,5]
        
        # Final classification
        return combined_features


class HADES_1S_2D(nn.Module):
    def __init__(self, hidden_units=8, output_dim=4, kernel_size_convolution=3, 
                 kernel_size_maxpool=2, stride=1, padding=1,dropout=0.3,lastfeatures=4):
        super().__init__()
        
        # Shared feature extractor for all 5 histogram types
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=hidden_units,
                     kernel_size=3, stride=stride, padding=padding),
            # nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units*2,
                     kernel_size=3, stride=stride, padding=padding),
            # nn.BatchNorm2d(hidden_units*2),
            nn.ReLU(),
            nn.MaxPool2d(3),
            # nn.Dropout2d(dropout),
            
            # Block 2
            nn.Conv2d(hidden_units*2, hidden_units*4, 5, stride, padding),
            # nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.Conv2d(hidden_units*4, hidden_units*8, 5, stride, padding),
            # nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.MaxPool2d(5),
            # nn.Dropout2d(dropout),
            
            # # Block 3
            # nn.Conv2d(hidden_units*8, hidden_units*16, 3, stride, padding),
            # # nn.BatchNorm2d(hidden_units*16),
            # nn.ReLU(),
            # nn.Conv2d(hidden_units*16, hidden_units*32, 3, stride, padding),
            # # nn.BatchNorm2d(hidden_units*32),
            # nn.ReLU(),
            # nn.MaxPool2d(3),
            # # nn.AdaptiveMaxPool2d(output_size=1)  # Global average pooling
            # # nn.Dropout2d(dropout),
            
            # Block 4
            nn.Conv2d(hidden_units*8, hidden_units*16, kernel_size_convolution, stride, padding),
            # nn.BatchNorm2d(hidden_units*64),
            nn.ReLU(),
            nn.Conv2d(hidden_units*16, hidden_units*32, kernel_size_convolution, stride, padding),
            # nn.BatchNorm2d(hidden_units*128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=1)  # Global average pooling
            # nn.MaxPool2d(2)
        )
        
        dummy = torch.zeros(1, 1, 64, 64)  # (batch=1, channel=1, height=48, width=48)
        with torch.no_grad():
            feat = self.feature_extractor(dummy)
            total_features = feat.flatten(1).shape[1]


        
        # Final classifier that combines information a histogram
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_features, total_features//4),  # Intermediate layer
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(total_features//4, output_dim),
            # nn.ReLU()
        )

    def forward(self, x):
        # Extract each histogram (assuming x is a dictionary)
        x = x["RICH_CalsXY"].unsqueeze(1).float()  # shape: [batch,1,48,48]

        # for i, layer in enumerate(self.feature_extractor):
        #     # print(f"Before {layer}: {x.shape}")
        #     x = layer(x)
            # print(f"After {layer}: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
        
        # Extract features from each histogram
        x = self.feature_extractor(x)   # shape: [batch, 1]
        # print(f"First exit shape: {x.shape}")
        x=self.classifier(x)
        # print(f"After classifier: {x.shape}")

        return x.squeeze()


class CNN_Model(L.LightningModule):
    def __init__(self,lr,hidden_units,kernel_size_convolution,kernel_size_maxpool,stride,padding,output_dim,version=1,input_size=50,what_dim="1D"):
        super().__init__()
        self.version=version
        self.what_dim=what_dim
        if self.what_dim=="1D":
            self.keys=["START_beam_mod1","START_beam_mod2","RICH_trends","ToF_multiplicity","ToF_sum"]
        elif self.what_dim=="2D":
            self.keys="RICH_CalsXY"
        else:
            print("What dims do we have now for those histograms?")
        self.lr=lr
        self.input_size=input_size
        self.hidden_units=hidden_units
        self.output_dim=output_dim
        self.kernel_size_convolution=kernel_size_convolution
        self.kernel_size_maxpool=kernel_size_maxpool
        self.stride=stride
        self.padding=padding




        self.save_hyperparameters()

        if self.version==0: #doesn't exist here!
            self.CNN=PaddingTon_9_Model_V0()
        if self.version==1:
            self.CNN=PaddingTon_9_Model_V1(hidden_units=self.hidden_units,
                                            kernel_size_convolution=self.kernel_size_convolution,
                                            kernel_size_maxpool=self.kernel_size_maxpool,
                                            stride=self.stride,
                                            padding=self.padding,
                                            output_dim=self.output_dim,
                                            input_size=self.input_size)
        elif self.version==2:
            self.CNN=HADES_5S_1D(hidden_units=self.hidden_units,
                                kernel_size_convolution=self.kernel_size_convolution,
                                kernel_size_maxpool=self.kernel_size_maxpool,
                                stride=self.stride,
                                padding=self.padding,
                                output_dim=self.output_dim)
        elif self.version==3:
            self.CNN=HADES_1S_2D(hidden_units=self.hidden_units,
                                kernel_size_convolution=self.kernel_size_convolution,
                                kernel_size_maxpool=self.kernel_size_maxpool,
                                stride=self.stride,
                                padding=self.padding,
                                output_dim=self.output_dim)

        self.metric = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        return self.CNN(inputs)

    def training_step(self, batch, batch_idx):
        if self.version==2:
            inputs,target=batch
            alllabel=[]
            for i in self.keys:
                alllabel.append(target[i].unsqueeze(1))
            target=torch.cat(alllabel,dim=1)

            output = self.forward(inputs)

            total_loss = 0
            for i in range(5):  # For each detector
                # print(f"in shape:{output[:,:,i].shape}")
                loss = self.metric(output[:,:,i], target[:,i])  # Compare with true labels
                total_loss += loss
                acc=hf.accuracy_fn(target[:,i],output[:,:,i].argmax(dim=1))
                self.log(f"train_acc {self.keys[i]}", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    
            loss= total_loss / 5  # Average loss across detectors
        elif self.version==3:
            inputs,target=batch
            target=target[self.keys]

            output = self.forward(inputs)
            loss = self.metric(output, target)  # Compare with true labels
            acc=hf.accuracy_fn(target,output.argmax(dim=1))
            
            self.log(f"train_acc {self.keys}", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        else:
            inputs, target = batch
            output = self.forward(inputs)
            loss =  self.metric(output, target)
            acc=hf.accuracy_fn(target,output.argmax(dim=1))
            self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.version==2:
            inputs,target=batch
            alllabel=[]
            for i in self.keys:
                alllabel.append(target[i].unsqueeze(1))
            target=torch.cat(alllabel,dim=1)

            output = self.forward(inputs)

            total_loss = 0
            for i in range(5):  # For each detector
                loss = self.metric(output[:,:,i], target[:,i])  # Compare with true labels
                total_loss += loss
                acc=hf.accuracy_fn(target[:,i],output[:,:,i].argmax(dim=1))
                self.log(f"val_acc {self.keys[i]}", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    
            loss= total_loss / 5  # Average loss across detectors
        elif self.version==3:
            inputs,target=batch
            target=target[self.keys]

            output = self.forward(inputs)
            loss = self.metric(output, target)  # Compare with true labels
            acc=hf.accuracy_fn(target,output.argmax(dim=1))
            
            self.log(f"val_acc {self.keys}", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        else:
            inputs, target = batch
            output = self.forward(inputs)
            loss =  self.metric(output, target)
            acc=hf.accuracy_fn(target,output.argmax(dim=1))
            self.log("val_acc", acc, logger=True)

        self.log("val_loss", loss, logger=True)


    def test_step(self, batch, batch_idx):
        if batch_idx==0:
                # self.deviations=torch.tensor([],device=self.device)
                self.targets=torch.tensor([],device=self.device)
                self.output=torch.tensor([],device=self.device)
        if self.version==2:
            inputs,target=batch
            alllabel=[]
            for i in self.keys:
                alllabel.append(target[i].unsqueeze(1))
            target=torch.cat(alllabel,dim=1)

            output = self.forward(inputs)

            self.output=torch.cat([self.output,output.detach()])
            self.targets=torch.cat([self.targets,target.detach()])

            total_loss = 0
            for i in range(5):  # For each detector
                loss = self.metric(output[:,:,i], target[:,i])  # Compare with true labels
                total_loss += loss
                acc=hf.accuracy_fn(target[:,i],output[:,:,i].argmax(dim=1))
                self.log(f"test_acc {self.keys[i]}", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    
            loss= total_loss / 5  # Average loss across detectors
        elif self.version==3:
            inputs,target=batch
            target=target[self.keys]

            output = self.forward(inputs)
            loss = self.metric(output, target)  # Compare with true labels
            acc=hf.accuracy_fn(target,output.argmax(dim=1))

            self.output=torch.cat([self.output,output.detach()])
            self.targets=torch.cat([self.targets,target.detach()])
            
            self.log(f"test_acc {self.keys}", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        else:
            inputs, target = batch
            output = self.forward(inputs)
            loss =  self.metric(output, target)
            acc=hf.accuracy_fn(target,output.argmax(dim=1))

            # self.deviations=torch.cat([self.deviations,(target-output).detach()])
            self.output=torch.cat([self.output,output.detach()])
            self.targets=torch.cat([self.targets,target.detach()])
            self.log("test_acc", acc)

        self.log("test_loss", loss)


    def on_test_epoch_end(self):
        if self.version==2:
            log_dir = self.trainer.log_dir or "logs"  # Fallback to "logs" if no logger
            os.makedirs(log_dir, exist_ok=True)

            targets=self.targets.cpu().numpy()
            csv_path_target = os.path.join(log_dir, "targets.csv")
            pd.DataFrame(targets, columns=[self.keys]).to_csv(csv_path_target, index=False)

            outputs_max=torch.softmax(self.output,dim=1).argmax(dim=1).cpu().numpy()
            csv_path_output = os.path.join(log_dir, "outputs_max.csv")
            pd.DataFrame(outputs_max, columns=[self.keys]).to_csv(csv_path_output, index=False)
            np.save(log_dir+"/outputs.npy",self.output.cpu().numpy())
        elif self.version==3:
            log_dir = self.trainer.log_dir or "logs"  # Fallback to "logs" if no logger
            os.makedirs(log_dir, exist_ok=True)

            targets=self.targets.cpu().numpy()
            csv_path_target = os.path.join(log_dir, "targets.csv")
            pd.DataFrame(targets, columns=[self.keys]).to_csv(csv_path_target, index=False)

            outputs_max=self.output.argmax(dim=1).cpu().numpy()
            csv_path_output = os.path.join(log_dir, "outputs_max.csv")
            pd.DataFrame(outputs_max, columns=[self.keys]).to_csv(csv_path_output, index=False)
            np.save(log_dir+"/outputs.npy",self.output.cpu().numpy())
        else:
            # Skip if no deviations collected
            # if not hasattr(self, 'deviations') or len(self.deviations) == 0:
            #     return

            # Concatenate all deviations (handles any tensor shape)
            # deviations = self.deviations.cpu().numpy()
            
            # Get the log directory (works with any logger: TensorBoard, CSV, etc.)
            log_dir = self.trainer.log_dir or "logs"  # Fallback to "logs" if no logger
            os.makedirs(log_dir, exist_ok=True)
            
            # Save to CSV in the log folder
            # csv_path_dev = os.path.join(log_dir, "deviations.csv")
            # pd.DataFrame(deviations, columns=["Deviation"]).to_csv(csv_path_dev, index=False)

            targets=self.targets.cpu().numpy()
            csv_path_target = os.path.join(log_dir, "targets.csv")
            pd.DataFrame(targets, columns=["Target"]).to_csv(csv_path_target, index=False)

            outputs=torch.softmax(self.output,dim=1).cpu().numpy()
            csv_path_output = os.path.join(log_dir, "outputs.csv")
            pd.DataFrame(outputs, columns=['none','empty_bin','skew','secondary_peak','noisy_bin']).to_csv(csv_path_output, index=False)
            
            # Optional: Reset for future test runs
            # self.deviations = []


    def configure_optimizers(self):
        #return torch.optim.SGD(self.model.parameters(), lr=0.1)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        
        return [optimizer]

class HADES_VAE_Encoder_5S_1D(nn.Module):
    def __init__(self, hidden_units=8, latent_dim=16, kernel_size_convolution=3, 
                 kernel_size_maxpool=2, stride=1, padding=1):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Shared feature extractor (same as before)
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=hidden_units,
                     kernel_size=kernel_size_convolution, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units*2,
                     kernel_size=kernel_size_convolution, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size_maxpool),
            
            # Block 2
            nn.Conv1d(hidden_units*2, hidden_units*4, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.Conv1d(hidden_units*4, hidden_units*6, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size_maxpool),
            
            # Block 3
            nn.Conv1d(hidden_units*6, hidden_units*8, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.Conv1d(hidden_units*8, hidden_units*10, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size_maxpool),
            
            # Block 4
            nn.Conv1d(hidden_units*10, hidden_units*12, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.Conv1d(hidden_units*12, hidden_units*14, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1)  # Global average pooling
        )
        
        # Calculate total features
        total_features = hidden_units * 14
        
        # Separate heads for mean and log-variance
        self.fc_mu = nn.Linear(total_features * 5, latent_dim)  # For all 5 histograms
        self.fc_logvar = nn.Linear(total_features * 5, latent_dim)

    def forward(self, x):
        # Extract features from each histogram
        features1 = self.feature_extractor(x["START_beam_mod1"].unsqueeze(1).float())
        features2 = self.feature_extractor(x["START_beam_mod2"].unsqueeze(1).float())
        features3 = self.feature_extractor(x["RICH_trends"].unsqueeze(1).float())
        features4 = self.feature_extractor(x["ToF_multiplicity"].unsqueeze(1).float())
        features5 = self.feature_extractor(x["ToF_sum"].unsqueeze(1).float())
        
        # Flatten and concatenate
        features1 = torch.flatten(features1, start_dim=1)
        features2 = torch.flatten(features2, start_dim=1)
        features3 = torch.flatten(features3, start_dim=1)
        features4 = torch.flatten(features4, start_dim=1)
        features5 = torch.flatten(features5, start_dim=1)
        
        combined = torch.cat([features1, features2, features3, features4, features5], dim=1)
        
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class HADES_VAE_Decoder_5S_1D(nn.Module):
    def __init__(self, latent_dim=32, hidden_units=8, output_shapes=None):
        super().__init__()
        
        # Default output shapes for each histogram
        if output_shapes is None:
            output_shapes = {
                "START_beam_mod1": 21,
                "START_beam_mod2": 21,
                "RICH_trends": 51,
                "ToF_multiplicity": 101,
                "ToF_sum": 151
            }
        
        self.output_shapes = output_shapes
        total_features = hidden_units * 14  # must match encoder output

        # Project latent vector back into feature map
        self.fc = nn.Linear(latent_dim, total_features * 5)  # one chunk per histogram

        # Build decoders for each histogram type
        self.decoders = nn.ModuleDict({
            'START_beam_mod1': self._build_decoder(total_features, output_shapes["START_beam_mod1"]),
            'START_beam_mod2': self._build_decoder(total_features, output_shapes["START_beam_mod2"]),
            'RICH_trends': self._build_decoder(total_features, output_shapes["RICH_trends"]),
            'ToF_multiplicity': self._build_decoder(total_features, output_shapes["ToF_multiplicity"]),
            'ToF_sum': self._build_decoder(total_features, output_shapes["ToF_sum"]),
        })

    def _build_decoder(self, channels, target_length):
            layers = []
            current_length = 1
            
            # Calculate required upsampling steps
            while current_length < target_length:
                layers.extend([
                    nn.ConvTranspose1d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(channels),
                ])
                current_length = current_length * 2
            
            # Adjust final layers to match exact target length
            if current_length > target_length:
                # Add adaptive pooling to get exact size
                layers.append(nn.AdaptiveAvgPool1d(target_length))
            
            # Final projection to 1 channel
            layers.extend([
                nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(channels // 2, 1, kernel_size=3, padding=1),
                nn.Flatten(start_dim=1)
            ])
            
            return nn.Sequential(*layers)

    def forward(self, z):
        # Project latent vector to feature space
        x = self.fc(z)  # [B, total_features*5]
        
        # Split into 5 parts, one per histogram
        chunks = torch.chunk(x, 5, dim=1)

        reconstructions = {}
        for i, (name, decoder) in enumerate(self.decoders.items()):
            h = chunks[i]  # get the i-th chunk [B, total_features]
            h = h.unsqueeze(-1)  # [B, C, 1] for ConvTranspose1d
            reconstructions[name] = decoder(h)
        
        return reconstructions


class HADES_VAE_Model(L.LightningModule):
    def __init__(self, lr, hidden_units, kernel_size_convolution, kernel_size_maxpool, 
                 stride, padding, latent_dim=32, beta=10.0, version=1, what_dim="1D", 
                 use_l1_loss=False, use_poisson_loss=False):
        super().__init__()
        self.version = version
        self.what_dim = what_dim
        if self.what_dim == "1D":
            self.keys = ["START_beam_mod1", "START_beam_mod2", "RICH_trends", "ToF_multiplicity", "ToF_sum"]
        elif self.what_dim == "2D":
            self.keys = ["RICH_CalsXY"]
        
        self.lr = lr
        self.hidden_units = hidden_units
        self.kernel_size_convolution = kernel_size_convolution
        self.kernel_size_maxpool = kernel_size_maxpool
        self.stride = stride
        self.padding = padding
        self.latent_dim = latent_dim
        self.beta = beta  # Higher beta for better regularization
        self.use_l1_loss = use_l1_loss
        self.use_poisson_loss = use_poisson_loss
        
        # Output shapes for each histogram type
        if self.what_dim == "1D":
            self.output_shapes = {
                "START_beam_mod1": 21,
                "START_beam_mod2": 21,
                "RICH_trends": 51,
                "ToF_multiplicity": 101,
                "ToF_sum": 151
            }
        
        self.save_hyperparameters()
        
        # Initialize VAE components
        if version == 1:
            self.encoder = HADES_VAE_Encoder_5S_1D(
                hidden_units=hidden_units,
                latent_dim=latent_dim,
                kernel_size_convolution=kernel_size_convolution,
                kernel_size_maxpool=kernel_size_maxpool,
                stride=stride,
                padding=padding
            )
            
            self.decoder = HADES_VAE_Decoder_5S_1D(
                latent_dim=latent_dim,
                hidden_units=hidden_units,
                output_shapes=self.output_shapes
            )

    def forward(self, inputs):
        mu, logvar = self.encoder(inputs)
        z = self.encoder.reparameterize(mu, logvar)
        reconstructions = self.decoder(z)
        return reconstructions, mu, logvar

    def _compute_reconstruction_loss(self, reconstruction, target):
        if self.use_poisson_loss:
            # For count data (histograms)
            return F.poisson_nll_loss(reconstruction, target, reduction='mean')
        elif self.use_l1_loss:
            # L1 loss for sparse reconstruction
            return F.l1_loss(reconstruction, target, reduction='mean')
        else:
            # Standard MSE
            return F.mse_loss(reconstruction, target, reduction='mean')

    def _compute_loss(self, batch, prefix="train"):
        inputs, _ = batch
        
        # Forward pass
        reconstructions, mu, logvar = self.forward(inputs)
        
        # Reconstruction loss for each histogram
        recon_loss = 0
        individual_losses = {}
        for key in self.keys:
            key_loss = self._compute_reconstruction_loss(reconstructions[key], inputs[key])
            recon_loss += key_loss
            individual_losses[key] = key_loss
        
        # KL divergence (properly normalized)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        # Logging
        self.log(f"{prefix}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{prefix}_recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{prefix}_kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        # Log individual reconstruction losses
        for key, loss_val in individual_losses.items():
            self.log(f"{prefix}_mse_{key}", loss_val, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return total_loss, reconstructions, mu

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._compute_loss(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._compute_loss(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.all_reconstructions = {key: [] for key in self.keys}
            self.all_inputs = {key: [] for key in self.keys}
            self.all_latents = []
        
        inputs, _ = batch
        loss, reconstructions, mu = self._compute_loss(batch, "test")
        
        # Store for later analysis
        for key in self.keys:
            self.all_reconstructions[key].append(reconstructions[key].detach().cpu())
            self.all_inputs[key].append(inputs[key].detach().cpu())
        self.all_latents.append(mu.detach().cpu())
        
        return loss

    def on_test_epoch_end(self):
        # Concatenate all test data
        reconstructions_concat = {}
        inputs_concat = {}
        
        for key in self.keys:
            reconstructions_concat[key] = torch.cat(self.all_reconstructions[key], dim=0)
            inputs_concat[key] = torch.cat(self.all_inputs[key], dim=0)
        
        latents_concat = torch.cat(self.all_latents, dim=0)
        
        # Get log directory
        log_dir = self.trainer.log_dir or "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Save reconstructions and original inputs
        for key in self.keys:
            np.save(os.path.join(log_dir, f"reconstructions_{key}.npy"), 
                   reconstructions_concat[key].numpy())
            np.save(os.path.join(log_dir, f"inputs_{key}.npy"), 
                   inputs_concat[key].numpy())
        
        # Save latent representations
        np.save(os.path.join(log_dir, "latent_vectors.npy"), latents_concat.numpy())
        
        # Save as CSV for easier inspection
        pd.DataFrame(latents_concat.numpy()).to_csv(
            os.path.join(log_dir, "latent_vectors.csv"), index=False
        )
        
        # Calculate and save final reconstruction errors
        mse_results = {}
        for key in self.keys:
            mse = F.mse_loss(reconstructions_concat[key], inputs_concat[key], reduction='mean')
            mse_results[key] = mse.item()
        
        pd.DataFrame.from_dict(mse_results, orient='index', columns=['MSE']).to_csv(
            os.path.join(log_dir, "reconstruction_errors.csv")
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=1e-5  # Added weight decay for regularization
        )
        
        # Learning rate scheduler with patience
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=8,
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }

    def generate_samples(self, num_samples=10):
        """Generate new samples from the learned latent space"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            generated = self.decoder(z)
            return generated

# class HADES_VAE_Model(L.LightningModule):
#     def __init__(self, lr, hidden_units, kernel_size_convolution, kernel_size_maxpool, 
#                  stride, padding, latent_dim=32, beta=1.0, version=1,what_dim="1D"):
#         super().__init__()
#         self.version = version
#         self.what_dim = what_dim
#         if self.what_dim=="1D":
#             self.keys = ["START_beam_mod1", "START_beam_mod2", "RICH_trends", "ToF_multiplicity", "ToF_sum"]
#         elif self.what_dim=="2D":
#             self.keys="RICH_CalsXY"
        
#         self.lr = lr
#         self.hidden_units = hidden_units
#         self.kernel_size_convolution = kernel_size_convolution
#         self.kernel_size_maxpool = kernel_size_maxpool
#         self.stride = stride
#         self.padding = padding
#         self.latent_dim = latent_dim
#         self.beta = beta  # Weight for KL divergence
        
#         # Output shapes for each histogram type
#         if self.what_dim=="1D":
#             self.output_shapes = {
#             "START_beam_mod1": 21,
#             "START_beam_mod2": 21,
#             "RICH_trends": 51,
#             "ToF_multiplicity": 101,
#             "ToF_sum": 151
#             }
#         elif self.what_dim=="2D":
#             # FIGURE OUT LATER
#             print("Whatever")
        
#         self.save_hyperparameters()
        
#         # Initialize VAE components
#         if version==1:
#             self.encoder = HADES_VAE_Encoder_5S_1D(
#                 hidden_units=hidden_units,
#                 latent_dim=latent_dim,
#                 kernel_size_convolution=kernel_size_convolution,
#                 kernel_size_maxpool=kernel_size_maxpool,
#                 stride=stride,
#                 padding=padding
#             )
            
#             # self.decoder = HADES_VAE_Decoder_5S_1D(
#             #     latent_dim=latent_dim,
#             #     hidden_units=hidden_units,
#             #     output_shapes=self.output_shapes
#             # )
#             self.decoder=HADES_VAE_Decoder_5S_1D(
#                 latent_dim=latent_dim,
#                 # base_channels=latent_dim,
#                 output_shapes=None
#             )
#         elif version!=1:
#             print("We're not ready for this yet!")

#     def forward(self, inputs):
#         mu, logvar = self.encoder(inputs)
#         z = self.encoder.reparameterize(mu, logvar)
#         reconstructions = self.decoder(z)
#         return reconstructions, mu, logvar

#     def _compute_loss(self, batch, prefix="train"):
#         inputs, _ = batch  # We don't need labels for unsupervised learning
        
#         # Forward pass
#         reconstructions, mu, logvar = self.forward(inputs)
        
#         # Reconstruction loss (MSE for each histogram)
#         recon_loss = 0
#         for key in self.keys:
#             recon_loss += F.mse_loss(reconstructions[key], inputs[key], reduction='mean')
        
#         # KL divergence
#         kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
#         # Total loss
#         total_loss = recon_loss + self.beta * kl_loss
        
#         # Logging
#         self.log(f"{prefix}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log(f"{prefix}_recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
#         self.log(f"{prefix}_kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
#         # Log reconstruction MSE for each histogram type
#         for key in self.keys:
#             mse = F.mse_loss(reconstructions[key], inputs[key], reduction='mean')
#             self.log(f"{prefix}_mse_{key}", mse, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
#         return total_loss, reconstructions, mu

#     def training_step(self, batch, batch_idx):
#         loss, _, _ = self._compute_loss(batch, "train")
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss, _, _ = self._compute_loss(batch, "val")
#         return loss

#     def test_step(self, batch, batch_idx):
#         if batch_idx == 0:
#             self.all_reconstructions = {key: [] for key in self.keys}
#             self.all_inputs = {key: [] for key in self.keys}
#             self.all_latents = []
        
#         inputs, _ = batch
#         loss, reconstructions, mu = self._compute_loss(batch, "test")
        
#         # Store for later analysis
#         for key in self.keys:
#             self.all_reconstructions[key].append(reconstructions[key].detach().cpu())
#             self.all_inputs[key].append(inputs[key].detach().cpu())
#         self.all_latents.append(mu.detach().cpu())
        
#         return loss

#     def on_test_epoch_end(self):
#         # Concatenate all test data
#         reconstructions_concat = {}
#         inputs_concat = {}
        
#         for key in self.keys:
#             reconstructions_concat[key] = torch.cat(self.all_reconstructions[key], dim=0)
#             inputs_concat[key] = torch.cat(self.all_inputs[key], dim=0)
        
#         latents_concat = torch.cat(self.all_latents, dim=0)
        
#         # Get log directory
#         log_dir = self.trainer.log_dir or "logs"
#         os.makedirs(log_dir, exist_ok=True)
        
#         # Save reconstructions and original inputs
#         for key in self.keys:
#             np.save(os.path.join(log_dir, f"reconstructions_{key}.npy"), 
#                    reconstructions_concat[key].numpy())
#             np.save(os.path.join(log_dir, f"inputs_{key}.npy"), 
#                    inputs_concat[key].numpy())
        
#         # Save latent representations
#         np.save(os.path.join(log_dir, "latent_vectors.npy"), latents_concat.numpy())
        
#         # Save as CSV for easier inspection (first few dimensions)
#         pd.DataFrame(latents_concat.numpy()).to_csv(
#             os.path.join(log_dir, "latent_vectors.csv"), index=False
#         )
        
#         # Calculate and save final reconstruction errors
#         mse_results = {}
#         for key in self.keys:
#             mse = F.mse_loss(reconstructions_concat[key], inputs_concat[key], reduction='mean')
#             mse_results[key] = mse.item()
        
#         pd.DataFrame.from_dict(mse_results, orient='index', columns=['MSE']).to_csv(
#             os.path.join(log_dir, "reconstruction_errors.csv")
#         )

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
#         # Optional: Add learning rate scheduler
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.5, patience=5
#         )
        
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "monitor": "val_loss",
#                 "frequency": 1
#             }
#         }

#     def generate_samples(self, num_samples=10):
#         """Generate new samples from the learned latent space"""
#         with torch.no_grad():
#             # Sample from standard normal
#             z = torch.randn(num_samples, self.latent_dim, device=self.device)
#             # Decode to generate samples
#             generated = self.decoder(z)
#             return generated




# class HADES_VAE_Decoder_5S_1D(nn.Module):
#     def __init__(self, latent_dim=32, hidden_units=8, output_shapes=None):
#         super().__init__()
        
#         # Output shapes for each histogram type
#         if output_shapes is None:
#             output_shapes = {
#                 "START_beam_mod1": 21,
#                 "START_beam_mod2": 21,
#                 "RICH_trends": 51,
#                 "ToF_multiplicity": 101,
#                 "ToF_sum": 151
#             }
        
#         self.output_shapes = output_shapes
#         total_features = hidden_units * 14
        
#         # Project latent vector back to feature space
#         self.fc = nn.Linear(latent_dim, total_features * 5)
        
#         # Separate decoders for each histogram type
#         self.decoders = nn.ModuleDict({
#             'START1': self._build_decoder(total_features, output_shapes["START_beam_mod1"]),
#             'START2': self._build_decoder(total_features, output_shapes["START_beam_mod2"]),
#             'RICH': self._build_decoder(total_features, output_shapes["RICH_trends"]),
#             'TOF_MULT': self._build_decoder(total_features, output_shapes["ToF_multiplicity"]),
#             'TOF_SUM': self._build_decoder(total_features, output_shapes["ToF_sum"])
#         })
    
#     def _build_decoder(self, input_dim, output_length):
#         return nn.Sequential(
#                 nn.Linear(input_dim, input_dim*16),
#                 nn.ReLU(),
#                 nn.Linear(input_dim*16,output_length),
#             )
    
#     def forward(self, z):
#         # Project latent vector
#         x = self.fc(z)
        
#         # Split for each histogram
#         chunks = torch.chunk(x, 5, dim=1)
        
#         # Reconstruct each histogram
#         reconstructions = {
#             "START_beam_mod1": self.decoders['START1'](chunks[0]),
#             "START_beam_mod2": self.decoders['START2'](chunks[1]),
#             "RICH_trends": self.decoders['RICH'](chunks[2]),
#             "ToF_multiplicity": self.decoders['TOF_MULT'](chunks[3]),
#             "ToF_sum": self.decoders['TOF_SUM'](chunks[4])
#         }
        
#         return reconstructions

class HADES_Autoencoder(nn.Module):
    def __init__(self, lr, hidden_units, kernel_size_convolution, kernel_size_maxpool, 
                 stride, padding, input_dims=[21, 21, 51, 101, 151]):
        super().__init__()
        
        self.keys = ["START_beam_mod1", "START_beam_mod2", "RICH_trends", 
                    "ToF_multiplicity", "ToF_sum"]
        self.lr = lr
        self.hidden_units = hidden_units
        self.input_dims = input_dims

        # Shared encoder
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=hidden_units,
        #              kernel_size=kernel_size_convolution, stride=stride, padding=padding),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units*2,
        #              kernel_size=kernel_size_convolution, stride=stride, padding=padding),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size_maxpool),
            
        #     nn.Conv1d(hidden_units*2, hidden_units*4, kernel_size_convolution, stride, padding),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size_maxpool),
        # )

        self.encoder=nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=hidden_units,
                     kernel_size=kernel_size_convolution, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units*2,
                     kernel_size=kernel_size_convolution, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size_maxpool),
            
            # Block 2
            nn.Conv1d(hidden_units*2, hidden_units*4, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.Conv1d(hidden_units*4, hidden_units*8, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size_maxpool),
            
            # Block 3
            nn.Conv1d(hidden_units*8, hidden_units*16, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.Conv1d(hidden_units*16, hidden_units*32, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size_maxpool),
            
            # Block 4
            nn.Conv1d(hidden_units*32, hidden_units*64, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.Conv1d(hidden_units*64, hidden_units*128, kernel_size_convolution, stride, padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1)  # Global average pooling
        )
        
        # Calculate the output size of the encoder
        # self.encoder_output_size = hidden_units * 4
        self.encoder_output_size = hidden_units * 128
        
        # Individual decoders with symmetric architecture
        self.decoders = nn.ModuleList()
        for output_dim in input_dims:
            decoder = nn.Sequential(
                nn.Linear(self.encoder_output_size, self.encoder_output_size//2),
                nn.ReLU(),
                nn.Linear(self.encoder_output_size//2, output_dim)
            )
            self.decoders.append(decoder)
    
    def forward(self, x):
        reconstructions = {}
        latent_features = []
        
        for i, hist_name in enumerate(self.keys):
            hist = x[hist_name].unsqueeze(1).float()
            encoded = self.encoder(hist).mean(dim=-1)  # Global average pooling
            decoded = self.decoders[i](encoded)
            
            reconstructions[hist_name] = decoded
            latent_features.append(encoded)
        
        # Concatenate all latent features
        combined_features = torch.cat(latent_features, dim=1)
        
        return reconstructions, combined_features, latent_features

class HADES_VAE_OTHER(L.LightningModule):
    def __init__(self, version,lr, hidden_units, kernel_size_convolution, kernel_size_maxpool, 
                 stride, padding, input_dims=[21, 21, 51, 101, 151]):
        super().__init__()
        self.lr = lr
        self.hidden_units = hidden_units 
        self.kernel_size_convolution = kernel_size_convolution
        self.kernel_size_maxpool = kernel_size_maxpool
        self.stride = stride
        self.padding = padding

        if version==2:
            self.model = HADES_Autoencoder(
                lr=self.lr, 
                hidden_units=self.hidden_units, 
                kernel_size_convolution=self.kernel_size_convolution, 
                kernel_size_maxpool=self.kernel_size_maxpool,
                stride=self.stride, 
                padding=self.padding
            )
        else:
            print("Model type not supported!")
        
        self.keys = ["START_beam_mod1", "START_beam_mod2", "RICH_trends", 
                    "ToF_multiplicity", "ToF_sum"]
        
        # Storage for latent features and labels
        self.latent_features = {key: [] for key in self.keys}
        self.labels = {key: [] for key in self.keys}  # Separate labels for each detector
        self.event_ids = []

        self.save_hyperparameters()

    def forward(self, inputs):
        return self.model(inputs)

    def compute_reconstruction_loss(self, reconstructions, original):
        """Compute reconstruction loss"""
        total_loss = 0
        for hist_name in self.keys:
            total_loss += F.mse_loss(reconstructions[hist_name], original[hist_name].float())
        return total_loss / len(self.keys)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        
        reconstructions, _, latent_features = self.forward(inputs)
        loss = self.compute_reconstruction_loss(reconstructions, inputs)
        
        self.log("train_loss",loss,on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        
        reconstructions, _, latent_features = self.forward(inputs)
        loss = self.compute_reconstruction_loss(reconstructions, inputs)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """Save latent features at the end of training"""
        if self.latent_features[self.keys[0]]:  # Check if any features were collected
            self.save_latent_features()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        
        # Get reconstructions and latent features
        reconstructions, combined_features, per_detector_features = self.forward(inputs)
        
        # Store latent features and labels for each detector
        for i, key in enumerate(self.keys):
            self.latent_features[key].append(per_detector_features[i].cpu().detach().numpy())
            
            # Store the corresponding label for this detector
            if isinstance(labels, dict):
                # Labels are a dictionary with separate labels for each detector
                detector_label = labels[key].cpu().numpy()
            else:
                # Labels are a tensor - assume same label for all detectors in an event
                detector_label = labels.cpu().numpy()
            
            self.labels[key].append(detector_label)
        
        # Calculate reconstruction loss
        loss = self.compute_reconstruction_loss(reconstructions, inputs)
        self.log("test_loss", loss)
        
        return loss

    def on_test_end(self):
        """Save all latent features and labels after testing"""
        self.save_latent_features()

    def save_latent_features(self):
        """Save latent features to numpy files and labels to CSV for each detector"""
        # Create output directory
        output_dir =str(self.trainer.log_dir)+"/latent_features"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save latent features and labels for each detector
        for key in self.keys:
            if self.latent_features[key] and self.labels[key]:
                features = np.concatenate(self.latent_features[key], axis=0)
                labels_array = np.concatenate(self.labels[key], axis=0)
                
                # Save features
                np.save(f"{output_dir}/{key}_latent_features.npy", features)
                print(f"Latent features for {key} saved to {output_dir}/{key}_latent_features.npy")
                
                # Save labels for this detector
                labels_df = pd.DataFrame({
                    'event_id': range(len(labels_array)),
                    'label': labels_array
                })
                labels_df.to_csv(f"{output_dir}/{key}_labels.csv", index=False)
                print(f"Labels for {key} saved to {output_dir}/{key}_labels.csv")
        
        # Also save combined labels if they exist (for reference)
        if any(self.labels.values()):
            # Get the first detector's labels as reference
            first_key = self.keys[0]
            if self.labels[first_key]:
                combined_labels = np.concatenate(self.labels[first_key], axis=0)
                combined_df = pd.DataFrame({
                    'event_id': range(len(combined_labels)),
                    'label': combined_labels
                })
                combined_df.to_csv(f"{output_dir}/combined_labels.csv", index=False)
                print(f"Combined labels saved to {output_dir}/combined_labels.csv")
        
        print("All latent features and labels saved successfully!")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_latent_features(self, dataloader):
        """Convenience method to get latent features for a dataloader"""
        self.eval()
        latent_features = {key: [] for key in self.keys}
        labels_dict = {key: [] for key in self.keys}  # Separate labels for each detector
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                # Move inputs to appropriate device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get latent features
                reconstructions, combined_features, per_detector_features = self(inputs)
                
                # Store features and labels for each detector
                for i, key in enumerate(self.keys):
                    latent_features[key].append(per_detector_features[i].cpu().numpy())
                    
                    # Store the corresponding label for this detector
                    if isinstance(labels, dict):
                        detector_label = labels[key].cpu().numpy()
                    else:
                        detector_label = labels.cpu().numpy()
                    
                    labels_dict[key].append(detector_label)
                
                if batch_idx % 100 == 0:
                    print(f"Processed batch {batch_idx}")
        
        # Concatenate all batches for each detector
        for key in self.keys:
            latent_features[key] = np.concatenate(latent_features[key], axis=0)
            labels_dict[key] = np.concatenate(labels_dict[key], axis=0)
        
        return latent_features, labels_dict

# Example usage:
# model = HADES_VAE_Simplified(
#     lr=1e-3,
#     hidden_units=16,  # Reduced from original
#     kernel_size_convolution=3,
#     kernel_size_maxpool=2,
#     stride=1,
#     padding=1,
#     input_dims=[21, 21, 51, 101, 151]
# )