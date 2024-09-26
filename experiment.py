# Inspired from https://github.com/AntixK/PyTorch-VAE/blob/master/experiment.py

import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

# MRH:
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer  as mlb
from umap import UMAP
from sklearn.decomposition import PCA
#

class VAEXperiment1(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment1, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

        ### MRH:
        self.latent_samples = []
        self.all_labels=[]
        self.new_latent_samples=[]
        self.new_all_labels=[]
        self.latent_samples_z1=[]
        self.latent_samples_z2=[]
        self.latent_samples_z3=[]
        self.latent_labels=[]
        self.Train_counter=1
        self.Val_counter=1
        ###

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):

        real_img, labels = batch
        self.curr_device = real_img.device
        #
        #print('\n inside training_step labels= ',labels)
        #print('\nmrh inside training_step real_img type & shape= ',type(real_img), real_img.shape, labels.shape) # torch.Size([64, 3, 64, 64]) torch.Size([64, 40])
        #print('mrh inside training i counter= ',self.Train_counter) #  Maximum= 2543
        self.Train_counter +=1
        self.Val_counter = 0
        #
        self.latent_samples_z1=[] # or append z2, z3, or concatenate all
        self.latent_samples_z2=[]
        self.latent_samples_z3=[]
        self.latent_labels=[]
        #
        #

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        ## MRH:
        # Log the training loss and other metrics, ensuring only scalars are logged
        train_loss_logging = {
            key: val.item() if isinstance(val, torch.Tensor) else val
            for key, val in train_loss.items()
        }

        # MRH:
        # self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        self.log_dict(train_loss_logging, sync_dist=True)

        # MRH:
        if batch_idx % 100 == 0:
          print('mi_loss= ', train_loss['mi_loss'], 'mi_loss12= ', train_loss['mi_loss12'],'mi_loss13= ',  train_loss['mi_loss13'], 'mi_loss23= ', train_loss['mi_loss23'])

        return train_loss['total_loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        #print('mrh inside experiment batch type & shape= ',type(batch), len(batch)) # <class 'list'> 2
        real_img, labels = batch
        #print('\n labels= ',labels)
        #raise Exception("Sorry, no numbers below zero")
        #print('\n\n------------------------------------------')
        #print('\nmrh inside experiment real_img type & shape= ',type(real_img), real_img.shape, labels.shape) # torch.Size([64, 3, 64, 64]) torch.Size([64, 40])
        #print('mrh inside experiment i counter= ',self.Val_counter) # Maximum= 311
        self.Val_counter += 1
        self.Train_counter = 0

        #print
        #
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        ### MRH
        # # Debugging: Print the shape of each loss value
        # for key, val in val_loss.items():
        #     print(f"{key}: shape = {val.shape}")

        # # Attempt to log losses
        # try:
        #     self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        # except RuntimeError as e:
        #     print(f"Error logging value: {e}")
        #     # Optionally raise the error again to stop execution
        #     raise Exception(" Mohammad Reza Hasanabadi")
        ###


        # MRH: to solve float problem
        # Log only scalar values or summarize them

        # Ensure 'total_loss' is logged as 'val_loss'
        self.log('val_loss', val_loss['total_loss'], on_step=False, on_epoch=True, sync_dist=True)

        val_loss_logging = {
            f"val_{key}": val.item() if isinstance(val, torch.Tensor) else val
            for key, val in val_loss.items()
        }

        #self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.log_dict(val_loss_logging, sync_dist=True)

        ###
        #self.new_latent_samples=[]
        #self.new_all_labels=[]
        z = results[6] # # Sample from the latent space
        z1=z[0]
        z2=z[1]
        z3=z[2]
        #
        #print('\n inside validation_step z1, z2, z3 shapes ',z1.shape,z2.shape,z3.shape)
        #
        self.latent_samples_z1.append(z1) # or append z2, z3, or concatenate all
        self.latent_samples_z2.append(z2)
        self.latent_samples_z3.append(z3)
        self.latent_labels.append(labels)

        #self.new_all_labels=labels  # Collect labels
        #
        #self.latent_samples.append(z1)
        #self.all_labels.append(labels)
        #print('inside validation_step latent_samples.append z1 z2,z3 latent shapes ',len(self.latent_samples_z1),len(self.latent_samples_z2),len(self.latent_samples_z3),len(self.latent_labels))
        # Inceasing 1 1 1 1, 2 2 2 2, 3 3 3 3, ... , 312 312 312 312

    def on_validation_end(self) -> None:
        self.sample_images()
        # MRH:
        #self.mrh_TSNE()
        # self.mrh_TSNE2()
        self.mrh_TSNE3()
        self.mrh_UMAP()
        self.mrh_PCA()
        self.test_and_sample()  # Call the new method here
        #self.test_and_sample2()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir ,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir ,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims


    ### MRH:
    def mrh_TSNE(self):

        #print('\n mrh inside mrh_TSNE=  ', type(self.latent_samples_z1),len(self.latent_samples_z1), len(self.latent_samples_z2), len(self.latent_samples_z3))
        # <class 'list'> 10 10 10 for 10 samples
        #print('\n mrh inside mrh_TSNE self.latent_samples_z1[0] & [1] shape =  ', self.latent_samples_z1[0].shape,self.latent_samples_z1[0].shape)
        #  torch.Size([64, 128]) torch.Size([64, 128])

        # Concatenate the latent samples and labels for plotting
        # MRH: using torch.cat converts a list of tensors [311,64] to one tensor  with torch.Size([19968, 128])
        # MRH: using numpy() convert torch.Size([19968, 128])  to 'numpy.ndarray'> (19968, 128)
        latent_samples_z1_cat = torch.cat(self.latent_samples_z1).detach().cpu().numpy()
        latent_samples_z2_cat = torch.cat(self.latent_samples_z2).detach().cpu().numpy()
        latent_samples_z3_cat = torch.cat(self.latent_samples_z3).detach().cpu().numpy()
        latent_samples_cat = [latent_samples_z1_cat, latent_samples_z2_cat, latent_samples_z3_cat]
        #print('mrh latent_samples_z1_cat ... ', latent_samples_z1_cat.shape, latent_samples_z2_cat.shape,latent_samples_z3_cat.shape)
        #  (640, 128) (640, 128) (640, 128) for 10 samples

        latent_labels_cat = torch.cat(self.latent_labels).detach().cpu().numpy()
        #print('latent_labels_cat= ',latent_labels_cat.shape) # (640, 40)  for 10 samples
        single_labels = np.argmax(latent_labels_cat, axis=1)
        #print('mrh single_labels= ',single_labels.shape) # (640)  for 10 samples
        #raise Exception("Sorry, Mohammad Reza Hasanabadi")

           # Create a figure for subplots
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

        # Perform T-SNE on each latent variable
        for i, latent_data in enumerate(latent_samples_cat):
            #print('mrh TSNE enumerate i = ', i) # inceasign 1, 2, ...
            #print('mrh TSNE enumerate latent_data shape= ',latent_data.shape )  # (640, 128) for 10 samples
            tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
            latent_2d = tsne.fit_transform(latent_data)

            # Plot t-SNE for each latent variable
            scatter = axes[i].scatter(latent_2d[:, 0], latent_2d[:, 1], c=single_labels, cmap='viridis', alpha=0.5)
            axes[i].set_title(f"T-SNE Visualization of z{i + 1} Latent Variables")
            axes[i].set_xlabel("t-SNE Component 1")
            axes[i].set_ylabel("t-SNE Component 2")
            axes[i].set_xlim(latent_2d[:, 0].min() - 1, latent_2d[:, 0].max() + 1)
            axes[i].set_ylim(latent_2d[:, 1].min() - 1, latent_2d[:, 1].max() + 1)
            # Combine all latent samples for a final visualization

        combined_latent = np.concatenate([latent_samples_z1_cat, latent_samples_z2_cat, latent_samples_z3_cat],  axis=0)
        #print('mrh  combined_latent',combined_latent.shape) # mrh  combined_latent (1920, 128)

        # Create labels corresponding to each sample for coloring
        num_z1 = len(latent_samples_z1_cat)
        num_z2 = len(latent_samples_z2_cat)
        num_z3 = len(latent_samples_z3_cat)

        # Create a color array that matches the total number of combined samples
        colors = np.array(['r'] * num_z1 + ['g'] * num_z2 + ['b'] * num_z3)
        #print('mrh colors= ',colors.shape )

        # Perform T-SNE on the combined latent variables
        combined_tsne = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(combined_latent)

        # Ensure the number of colors matches the number of combined samples
        assert len(colors) == combined_tsne.shape[0], f"Colors length: {len(colors)}, combined_tsne shape: {combined_tsne.shape}"

        # Plot the combined T-SNE representation
        scatter_combined = axes[3].scatter(combined_tsne[:, 0], combined_tsne[:, 1], c=colors, alpha=0.5)
        axes[3].set_title("Combined T-SNE Visualization of Latent Variables z1, z2, z3")
        axes[3].set_xlabel("t-SNE Component 1")
        axes[3].set_ylabel("t-SNE Component 2")
        axes[3].set_xlim(combined_tsne[:, 0].min() - 1, combined_tsne[:, 0].max() + 1)
        axes[3].set_ylim(combined_tsne[:, 1].min() - 1, combined_tsne[:, 1].max() + 1)

        # Adjust layout and add colorbar
        plt.colorbar(scatter_combined, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)
        plt.tight_layout()
        plt.show()

    def  mrh_TSNE2(self):

        latent_samples_z1_cat = torch.cat(self.latent_samples_z1).detach().cpu().numpy()
        latent_samples_z2_cat = torch.cat(self.latent_samples_z2).detach().cpu().numpy()
        latent_samples_z3_cat = torch.cat(self.latent_samples_z3).detach().cpu().numpy()
        #latent_samples_cat = [latent_samples_z1_cat, latent_samples_z2_cat, latent_samples_z3_cat]

        # Combine all latent samples into a single matrix
        combined_latent = np.concatenate([latent_samples_z1_cat, latent_samples_z2_cat, latent_samples_z3_cat], axis=0)

        # Create labels corresponding to each sample for coloring
        num_z1 = len(latent_samples_z1_cat)
        num_z2 = len(latent_samples_z2_cat)
        num_z3 = len(latent_samples_z3_cat)

        # Create a color array that matches the total number of combined samples
        colors = np.array(['r'] * num_z1 + ['g'] * num_z2 + ['b'] * num_z3)

        # Perform T-SNE on the combined latent variables
        combined_tsne = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(combined_latent)

        # Ensure the number of colors matches the number of combined samples
        assert len(colors) == combined_tsne.shape[0], f"Colors length: {len(colors)}, combined_tsne shape: {combined_tsne.shape}"

        # Create a figure and axes for plotting
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Individual T-SNE plots for z1, z2, z3
        axes[0, 0].scatter(combined_tsne[:num_z1, 0], combined_tsne[:num_z1, 1], c='red', alpha=0.5)
        axes[0, 0].set_title("T-SNE Visualization of z1 Latent Variables")
        axes[0, 0].set_xlabel("t-SNE Component 1")
        axes[0, 0].set_ylabel("t-SNE Component 2")
        #axes[0, 0].set_xlim(-5, 5)  # Adjust x-axis limits
        #axes[0, 0].set_ylim(-5, 5)  # Adjust y-axis limits

        axes[0, 1].scatter(combined_tsne[num_z1:num_z1+num_z2, 0], combined_tsne[num_z1:num_z1+num_z2, 1], c='green', alpha=0.5)
        axes[0, 1].set_title("T-SNE Visualization of z2 Latent Variables")
        axes[0, 1].set_xlabel("t-SNE Component 1")
        axes[0, 1].set_ylabel("t-SNE Component 2")
        #axes[0, 1].set_xlim(-5, 5)  # Adjust x-axis limits
        #axes[0, 1].set_ylim(-5, 5)  # Adjust y-axis limits

        axes[1, 0].scatter(combined_tsne[num_z1+num_z2:, 0], combined_tsne[num_z1+num_z2:, 1], c='blue', alpha=0.5)
        axes[1, 0].set_title("T-SNE Visualization of z3 Latent Variables")
        axes[1, 0].set_xlabel("t-SNE Component 1")
        axes[1, 0].set_ylabel("t-SNE Component 2")
        #axes[1, 0].set_xlim(-5, 5)  # Adjust x-axis limits
        #axes[1, 0].set_ylim(-5, 5)  # Adjust y-axis limits


        # Combined T-SNE representation
        scatter_combined = axes[1, 1].scatter(combined_tsne[:, 0], combined_tsne[:, 1], c=colors, alpha=0.5)
        axes[1, 1].set_title("Combined T-SNE Visualization of Latent Variables z1, z2, z3")
        axes[1, 1].set_xlabel("t-SNE Component 1")
        axes[1, 1].set_ylabel("t-SNE Component 2")
        #axes[1, 1].set_xlim(-5, 5)  # Adjust x-axis limits
        #axes[1, 1].set_ylim(-5, 5)  # Adjust y-axis limits


        # Adjust layout and add colorbar
        plt.colorbar(scatter_combined, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)
        plt.tight_layout()
        plt.show()
##
    def  mrh_TSNE3(self):

        latent_samples_z1_cat = torch.cat(self.latent_samples_z1).detach().cpu().numpy()
        latent_samples_z2_cat = torch.cat(self.latent_samples_z2).detach().cpu().numpy()
        latent_samples_z3_cat = torch.cat(self.latent_samples_z3).detach().cpu().numpy()
        #latent_samples_cat = [latent_samples_z1_cat, latent_samples_z2_cat, latent_samples_z3_cat]

        #
        latent_labels_cat = torch.cat(self.latent_labels).detach().cpu().numpy()
        class_indices = np.where(latent_labels_cat == 1)[1]  # Get indices of classes where label is 1

        # Combine all latent samples into a single matrix
        combined_latent = np.concatenate([latent_samples_z1_cat, latent_samples_z2_cat, latent_samples_z3_cat], axis=0)

        # Create labels corresponding to each sample for coloring
        num_z1 = len(latent_samples_z1_cat)
        num_z2 = len(latent_samples_z2_cat)
        num_z3 = len(latent_samples_z3_cat)

        # Create a color array that matches the total number of combined samples
        colors = np.array(['r'] * num_z1 + ['g'] * num_z2 + ['b'] * num_z3)

        # Perform T-SNE on the combined latent variables
        combined_tsne = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(combined_latent)

        # Ensure the number of colors matches the number of combined samples
        assert len(colors) == combined_tsne.shape[0], f"Colors length: {len(colors)}, combined_tsne shape: {combined_tsne.shape}"

        # Create a figure and axes for plotting
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Create a color map for the classes
        unique_classes = np.unique(class_indices)
        class_colors = plt.cm.tab10(np.arange(len(unique_classes)))  # Use distinct colors from the colormap
        class_color_map = {cls: class_colors[i] for i, cls in enumerate(unique_classes)}

        # Create a list of class names (you might need to customize this based on your actual classes)
        class_names = [f"Class {i}" for i in unique_classes]  # or replace with actual names if available


        for ax, latent_data, title, start_index, num_samples in zip(
            [axes[0, 0], axes[0, 1], axes[1, 0]],
            [latent_samples_z1_cat, latent_samples_z2_cat, latent_samples_z3_cat],
            ["T-SNE Visualization of z1 Latent Variables", "T-SNE Visualization of z2 Latent Variables", "T-SNE Visualization of z3 Latent Variables"],
            [0, len(latent_samples_z1_cat), len(latent_samples_z1_cat) + len(latent_samples_z2_cat)],
            [len(latent_samples_z1_cat), len(latent_samples_z2_cat), len(latent_samples_z3_cat)]
        ):
            ax.scatter(combined_tsne[start_index:start_index + num_samples, 0],
                      combined_tsne[start_index:start_index + num_samples, 1],
                      c=[class_color_map[cls] for cls in class_indices[start_index:start_index + num_samples]], alpha=0.5)
            ax.set_title(title)
            ax.set_xlabel("t-SNE Component 1")
            ax.set_ylabel("t-SNE Component 2")

        #
        # Create legend
        # handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[i],
        #                       markerfacecolor=class_color_map[i], markersize=10) for i in unique_classes]
        # ax.legend(handles=handles, title="Classes", loc='upper right')


        # Combined T-SNE representation
        scatter_combined = axes[1, 1].scatter(combined_tsne[:, 0], combined_tsne[:, 1], c=colors, alpha=0.5)
        axes[1, 1].set_title("Combined T-SNE Visualization of Latent Variables z1, z2, z3")
        axes[1, 1].set_xlabel("t-SNE Component 1")
        axes[1, 1].set_ylabel("t-SNE Component 2")
        #axes[1, 1].set_xlim(-5, 5)  # Adjust x-axis limits
        #axes[1, 1].set_ylim(-5, 5)  # Adjust y-axis limits


        # Adjust layout and add colorbar
        plt.colorbar(scatter_combined, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)
        plt.tight_layout()
        plt.show()
###
    def test_and_sample(self):
        # New method to sample from the latent space and generate images
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        #print('\n test_and_sample= ',test_input.shape, test_label.shape) # test_and_sample=  torch.Size([144, 3, 64, 64]) # torch.Size([144, 40])
        #print('test_label= ',test_label)
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        # Create the 'Generated_Samples' directory if it doesn't exist
        samples_dir = os.path.join(self.logger.log_dir, "Generated_Samples")
        os.makedirs(samples_dir, exist_ok=True)  # Create the directory if it doesn't exist


        # Get samples from the model
        generated_images = self.sample_latent_and_generate(test_input, test_label)
        #print('generated_images shape= ', generated_images.shape) # generated_images shape=  torch.Size([12, 3, 64, 64])

        # Save generated images
        vutils.save_image(generated_images.cpu().data,
                          os.path.join(self.logger.log_dir,
                                       "Generated_Samples",
                                       f"{self.logger.name}_Generated_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

    def sample_latent_and_generate(self, input_data, labels, num_samples=144):
        # Forward pass through the model to get mu and logvar
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            #print('input_data= ',input_data.shape) # input_data=  torch.Size([144, 3, 64, 64])
            results = self.forward(input_data, labels=labels)
            mu_q_list, log_var_q_list = results[2:4]  # Assuming mu and logvar are the 3rd and 4th outputs
            #print('mu_q_list, log_var_q_list len and [0].shape= ',len(mu_q_list),len(log_var_q_list),mu_q_list[0].shape,log_var_q_list[0].shape )
            # 3 3 torch.Size([144, 128]) torch.Size([144, 128])

        # Sampling from latent space for z1, z2, z3
        z_samples = []
        for i in range(3):  # Assuming you want to sample from z1, z2, z3
            mu = mu_q_list[i] # torch.Size([144, 128])
            log_var = log_var_q_list[i] # torch.Size([144, 128])
            sampled_z = self.model.reparameterize(mu, log_var)
            # print('sampled_z= ',sampled_z.shape) # torch.Size([144, 128])
            z_samples.append(sampled_z)

        # Decode the samples using the decoder (for the last layer z3)
        generated_images = self.model.decode(z_samples[2][:num_samples])  # Decode from z3

        return generated_images
##


    def test_and_sample2(self):
        # Other parts of the method remain mostly unchanged...

        # Load the test data
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        # Create directory for generated samples
        samples_dir = os.path.join(self.logger.log_dir, "Generated_Samples2")
        os.makedirs(samples_dir, exist_ok=True)

        # Generate a 12x12 grid of images based on varying latent dimensions
        generated_images_grid = self.generate_latent_grid(test_input, test_label)

        # Save the generated images grid
        vutils.save_image(generated_images_grid.cpu(),
                          os.path.join(samples_dir, f"{self.logger.name}_Grid_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)  # Since we want a grid of 12 images wide

    def generate_latent_grid(self, input_data, labels, grid_size=12):
        # Forward pass to get the mean and log variance
        self.model.eval()
        with torch.no_grad():
            results = self.forward(input_data, labels=labels)
            mu_q_list, log_var_q_list = results[2:4]

        # Sample from z1, z2, z3
        z1_samples = self.model.reparameterize(mu_q_list[0], log_var_q_list[0])  # Assuming z1 is at index 0
        z2_samples = self.model.reparameterize(mu_q_list[1], log_var_q_list[1])  # For z2
        z3_samples = self.model.reparameterize(mu_q_list[2], log_var_q_list[2])  # For z3

        # Check shapes
        #print(f"z1_samples shape: {z1_samples.shape}")  # Debugging line
        #print(f"z2_samples shape: {z2_samples.shape}")  # Debugging line
        #print(f"z3_samples shape: {z3_samples.shape}")  # Debugging line

        # Create the grid
        generated_images = []
        varying_z1_indices = [0, 1]  # Example indices (change as necessary)
        fixed_z1_indices = [2, 3]     # Example fixed indices (change as necessary)

        varying_values = torch.linspace(-3, 3, grid_size)  # Change this range as needed

        # Create the grid with varying and fixed latent values
        for row in range(grid_size):
            for col in range(grid_size):
                # Create a latent vector from the fixed z1 values
                latent_vector = torch.zeros(z1_samples.shape[1]).to(input_data.device)  # Assuming 128 dim

                # Set fixed values for z1
                for idx in fixed_z1_indices:
                    latent_vector[idx] = 0  # Set fixed dimension (can adjust to your needs)

                # Vary one dimension along the row and one along the column
                latent_vector[varying_z1_indices[0]] = varying_values[row]  # Row variation
                latent_vector[varying_z1_indices[1]] = varying_values[col]  # Column variation

                # Ensure that z2_samples and z3_samples have the expected shapes before accessing
                if z2_samples.numel() > 0:
                    latent_vector[z2_samples.shape[-1] - 1] = 0  # Assuming shape[-1] gives the last dimension size
                else:
                    print("Warning: z2_samples is empty. Skipping variation for z2.")

                if z3_samples.numel() > 0:
                    latent_vector[z3_samples.shape[-1] - 1] = 0  # Same check for z3
                else:
                    print("Warning: z3_samples is empty. Skipping variation for z3.")

                # Expand latent_vector to match the number of samples
                generated_images.append(latent_vector)

        # Convert to tensor and decode to images
        generated_images_tensor = torch.stack(generated_images)  # Shape: [144, latent_dim]
        decoded_images = self.model.decode(generated_images_tensor.detach())  # Ensure decoding is done properly

        return decoded_images  # Shape: [144, 3, 64, 64]

#
    def generate_latent_grid2(self, input_data, labels, grid_size=12):
        # Forward pass to get the mean and log variance
        self.model.eval()
        with torch.no_grad():
            results = self.forward(input_data, labels=labels)
            mu_q_list, log_var_q_list = results[2:4]

        # Sample from z1, z2, z3
        z1_samples = self.model.reparameterize(mu_q_list[0], log_var_q_list[0])
        z2_samples = self.model.reparameterize(mu_q_list[1], log_var_q_list[1])
        z3_samples = self.model.reparameterize(mu_q_list[2], log_var_q_list[2])

        # Check shapes
        #print(f"z1_samples shape: {z1_samples.shape}")
        #print(f"z2_samples shape: {z2_samples.shape}")
        #print(f"z3_samples shape: {z3_samples.shape}")

        # Create the grid
        generated_images = []
        varying_indices = [0, 1, 2]  # Indices for varying features (e.g., expression, age, gender)
        fixed_indices = [3, 4]        # Indices for fixed features

        varying_values = torch.linspace(-3, 3, grid_size)

        for row in range(grid_size):
            for col in range(grid_size):
                latent_vector = torch.zeros(z1_samples.shape[1]).to(input_data.device)

                # Set fixed values for certain features
                for idx in fixed_indices:
                    latent_vector[idx] = 0

                # Vary features based on row and column
                latent_vector[varying_indices[0]] = varying_values[row]  # Feature 1 variation
                latent_vector[varying_indices[1]] = varying_values[col]  # Feature 2 variation

                # Add more variations as needed
                # latent_vector[varying_indices[2]] = some_other_value  # For a third feature

                generated_images.append(latent_vector)

        generated_images_tensor = torch.stack(generated_images)
        decoded_images = self.model.decode(generated_images_tensor.detach())

        return decoded_images

##
    def generate_latent_grid3(self, input_data, labels, grid_size=12):
        # Forward pass to get the mean and log variance
        self.model.eval()
        with torch.no_grad():
            results = self.forward(input_data, labels=labels)
            mu_q_list, log_var_q_list = results[2:4]

        # Sample from different latent variables
        z_samples = [self.model.reparameterize(mu_q, log_var_q) for mu_q, log_var_q in zip(mu_q_list, log_var_q_list)]

        # Check shapes for each sampled latent variable
        for i, z in enumerate(z_samples):
            print(f"z{i+1}_samples shape: {z.shape}")

        # Create the grid
        generated_images = []
        varying_indices = {
            'expression': 0,
            'pose': 1,
            'accessory': 2,
            'skin_tone': 3,
            'lighting': 4,
            'hair_style': 5,
        }

        fixed_indices = [6, 7]  # Adjust based on your model's latent space

        # Define varying values for features
        varying_values = {key: torch.linspace(-3, 3, grid_size) for key in varying_indices.keys()}

        for row in range(grid_size):
            for col in range(grid_size):
                latent_vector = torch.zeros(z_samples[0].shape[1]).to(input_data.device)

                # Set fixed values for certain features
                for idx in fixed_indices:
                    latent_vector[idx] = 0

                # Vary features based on row and column
                latent_vector[varying_indices['expression']] = varying_values['expression'][row]
                latent_vector[varying_indices['pose']] = varying_values['pose'][col]

                # Depending on your experimentation design, you might want to vary other features
                latent_vector[varying_indices['accessory']] = varying_values['accessory'][row]  # Example usage
                latent_vector[varying_indices['skin_tone']] = varying_values['skin_tone'][col]

                generated_images.append(latent_vector)

        generated_images_tensor = torch.stack(generated_images)
        decoded_images = self.model.decode(generated_images_tensor.detach())

        return decoded_images
##
    def mrh_UMAP(self):
        # Retrieve the latent samples
        latent_samples_z1_cat = torch.cat(self.latent_samples_z1).detach().cpu().numpy()
        latent_samples_z2_cat = torch.cat(self.latent_samples_z2).detach().cpu().numpy()
        latent_samples_z3_cat = torch.cat(self.latent_samples_z3).detach().cpu().numpy()

        # Combine all latent samples into a single matrix
        combined_latent = np.concatenate([latent_samples_z1_cat, latent_samples_z2_cat, latent_samples_z3_cat], axis=0)

        # Create labels corresponding to each sample for coloring
        num_z1 = len(latent_samples_z1_cat)
        num_z2 = len(latent_samples_z2_cat)
        num_z3 = len(latent_samples_z3_cat)

        # Create a color array that matches the total number of combined samples
        colors = np.array(['r'] * num_z1 + ['g'] * num_z2 + ['b'] * num_z3)

        # Perform UMAP on the combined latent variables
        umap_embeddings = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(combined_latent)

        # Create a figure and axes for plotting
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Individual UMAP plots for z1, z2, z3
        axes[0, 0].scatter(umap_embeddings[:num_z1, 0], umap_embeddings[:num_z1, 1], c='red', alpha=0.5)
        axes[0, 0].set_title("UMAP Visualization of z1 Latent Variables")
        axes[0, 0].set_xlabel("UMAP Component 1")
        axes[0, 0].set_ylabel("UMAP Component 2")

        axes[0, 1].scatter(umap_embeddings[num_z1:num_z1+num_z2, 0], umap_embeddings[num_z1:num_z1+num_z2, 1], c='green', alpha=0.5)
        axes[0, 1].set_title("UMAP Visualization of z2 Latent Variables")
        axes[0, 1].set_xlabel("UMAP Component 1")
        axes[0, 1].set_ylabel("UMAP Component 2")

        axes[1, 0].scatter(umap_embeddings[num_z1+num_z2:, 0], umap_embeddings[num_z1+num_z2:, 1], c='blue', alpha=0.5)
        axes[1, 0].set_title("UMAP Visualization of z3 Latent Variables")
        axes[1, 0].set_xlabel("UMAP Component 1")
        axes[1, 0].set_ylabel("UMAP Component 2")

        # Combined UMAP representation
        scatter_combined = axes[1, 1].scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=colors, alpha=0.5)
        axes[1, 1].set_title("Combined UMAP Visualization of Latent Variables z1, z2, z3")
        axes[1, 1].set_xlabel("UMAP Component 1")
        axes[1, 1].set_ylabel("UMAP Component 2")

        # Adjust layout and add colorbar
        plt.colorbar(scatter_combined, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)
        plt.tight_layout()
        plt.show()

    def mrh_PCA(self):
        # Retrieve the latent samples
        latent_samples_z1_cat = torch.cat(self.latent_samples_z1).detach().cpu().numpy()
        latent_samples_z2_cat = torch.cat(self.latent_samples_z2).detach().cpu().numpy()
        latent_samples_z3_cat = torch.cat(self.latent_samples_z3).detach().cpu().numpy()

        # Combine all latent samples into a single matrix
        combined_latent = np.concatenate([latent_samples_z1_cat, latent_samples_z2_cat, latent_samples_z3_cat], axis=0)

        # Create labels corresponding to each sample for coloring
        num_z1 = len(latent_samples_z1_cat)
        num_z2 = len(latent_samples_z2_cat)
        num_z3 = len(latent_samples_z3_cat)

        # Create a color array that matches the total number of combined samples
        colors = np.array(['r'] * num_z1 + ['g'] * num_z2 + ['b'] * num_z3)

        # Perform PCA on the combined latent variables
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(combined_latent)

        # Create a figure and axes for plotting
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Individual PCA plots for z1, z2, z3
        axes[0, 0].scatter(pca_embeddings[:num_z1, 0], pca_embeddings[:num_z1, 1], c='red', alpha=0.5)
        axes[0, 0].set_title("PCA Visualization of z1 Latent Variables")
        axes[0, 0].set_xlabel("PCA Component 1")
        axes[0, 0].set_ylabel("PCA Component 2")

        axes[0, 1].scatter(pca_embeddings[num_z1:num_z1+num_z2, 0], pca_embeddings[num_z1:num_z1+num_z2, 1], c='green', alpha=0.5)
        axes[0, 1].set_title("PCA Visualization of z2 Latent Variables")
        axes[0, 1].set_xlabel("PCA Component 1")
        axes[0, 1].set_ylabel("PCA Component 2")

        axes[1, 0].scatter(pca_embeddings[num_z1+num_z2:, 0], pca_embeddings[num_z1+num_z2:, 1], c='blue', alpha=0.5)
        axes[1, 0].set_title("PCA Visualization of z3 Latent Variables")
        axes[1, 0].set_xlabel("PCA Component 1")
        axes[1, 0].set_ylabel("PCA Component 2")

        # Combined PCA representation
        scatter_combined = axes[1, 1].scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=colors, alpha=0.5)
        axes[1, 1].set_title("Combined PCA Visualization of Latent Variables z1, z2, z3")
        axes[1, 1].set_xlabel("PCA Component 1")
        axes[1, 1].set_ylabel("PCA Component 2")

        # Adjust layout
        plt.tight_layout()
        plt.show()
