import argparse
import torchvision
from torchvision import transforms
import torch
from torch import nn
import torch.optim as optim
import copy
import os
import kornia
from model import ImgFusNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device('cuda:1'))
    opt = parser.parse_args()
    batch_size = 32
    epochs = 600
    lr = 1e-4
    data_path = './datasets/data/'
    train_root_VIS = data_path + 'train_IR'
    train_root_IR = data_path + 'train_VI'
    val_root_VIS = data_path + 'val_IR'
    val_root_IR = data_path + 'val_VI'
    train_path = './output/'
    Train_Image_Number = len(os.listdir(train_root_VIS + '/train_IR_crop'))
    Val_Image_Number = len(os.listdir(val_root_VIS + '/val_IR_crop'))

    train_Iter_per_epoch = (Train_Image_Number % batch_size != 0) + Train_Image_Number // batch_size
    val_Iter_per_epoch = (Train_Image_Number % batch_size != 0) + Train_Image_Number // batch_size
    # =============================================================================
    # Preprocessing and dataset establishment
    # =============================================================================

    transforms = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])
    train_Data_VIS = torchvision.datasets.ImageFolder(train_root_VIS, transform=transforms)
    train_dataloader_VIS = torch.utils.data.DataLoader(train_Data_VIS, batch_size, shuffle=False)
    train_Data_IR = torchvision.datasets.ImageFolder(train_root_IR, transform=transforms)
    train_dataloader_IR = torch.utils.data.DataLoader(train_Data_IR, batch_size, shuffle=False)

    val_Data_VIS = torchvision.datasets.ImageFolder(train_root_VIS, transform=transforms)
    val_dataloader_VIS = torch.utils.data.DataLoader(val_Data_VIS, batch_size, shuffle=False)
    val_Data_IR = torchvision.datasets.ImageFolder(train_root_IR, transform=transforms)
    val_dataloader_IR = torch.utils.data.DataLoader(val_Data_IR, batch_size, shuffle=False)

    # =============================================================================
    # Models
    # =============================================================================
    Net = ImgFusNet().to(opt.device)
    optimizer1 = optim.Adam(Net.parameters(), lr=lr)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [epochs // 3, epochs // 3 * 2], gamma=0.1)
    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()
    ssim = kornia.losses.SSIMLoss(11, reduction='mean')
    # =============================================================================
    # Training
    # =============================================================================
    print('============ Training Begins ===============')

    lr_list1 = []

    best_weights = copy.deepcopy(Net.state_dict())
    best_epoch = 0
    # best_SAM = 5.1
    best_loss = 8

    for iteration in range(epochs):
        train_data_iter_VIS = iter(train_dataloader_VIS)
        train_data_iter_IR = iter(train_dataloader_IR)
        val_data_iter_VIS = iter(val_dataloader_VIS)
        val_data_iter_IR = iter(val_dataloader_IR)

        loss_train = []

        for step in range(train_Iter_per_epoch):
            data_VIS, _ = next(train_data_iter_VIS)
            data_IR, _ = next(train_data_iter_IR)

            data_VIS = data_VIS.to(opt.device)
            data_IR = data_IR.to(opt.device)
            optimizer1.zero_grad()
            # optimizer2.zero_grad()
            # =====================================================================
            # Calculate loss
            # =====================================================================
            data_fus, img_1, img_2, res1, res2 = Net(data_IR, data_VIS, 3)
            L2_ir_loss = torch.norm(data_IR - img_1)
            L2_vi_loss = torch.norm(data_VIS - img_2)
            Lrecon = L2_ir_loss + L2_vi_loss
            L1_loss = L1Loss(data_VIS, data_fus) + L1Loss(data_IR, data_fus)
            Lsp = torch.mean(abs(res1)) + torch.mean(abs(res2))
            Gradient_loss_IF = L1Loss(
                kornia.filters.SpatialGradient()(data_IR),
                kornia.filters.SpatialGradient()(data_fus)
            )
            Gradient_loss_VF = L1Loss(
                kornia.filters.SpatialGradient()(data_VIS),
                kornia.filters.SpatialGradient()(data_fus)
            )
            Lgra = Gradient_loss_IF + Gradient_loss_VF
            # Total loss
            train_loss = 2*Lrecon + 4*L1_loss + 4*Lgra + Lsp
            train_loss.backward()
            optimizer1.step()
            los = train_loss.item()
            loss_train.append(train_loss.item())
            if step % 100 == 0:
                print("step:", step, "/", Train_Image_Number//batch_size)
        train_epoch_loss = torch.mean(torch.tensor(loss_train))

        print('train total loss: {:.6f}'.format(train_epoch_loss))
        print('Epoch:%d,lr: %f' % (
            iteration + 1, optimizer1.state_dict()['param_groups'][0]['lr']))

        with torch.no_grad():
            loss_val = []
            for i in range(val_Iter_per_epoch):
                data_VIS, _ = next(val_data_iter_VIS)
                data_IR, _ = next(val_data_iter_IR)

                data_VIS = data_VIS.to(opt.device)
                data_IR = data_IR.to(opt.device)
                # Calculate loss
                # =====================================================================
                data_fus, img_1, img_2, res1, res2 = Net(data_IR, data_VIS, 3)
                L2_ir_loss = torch.norm(data_IR - img_1)
                L2_vi_loss = torch.norm(data_VIS - img_2)
                Lrecon = L2_ir_loss + L2_vi_loss
                L1_loss = L1Loss(data_VIS, data_fus) + L1Loss(data_IR, data_fus)
                Lsp = torch.mean(abs(res1)) + torch.mean(abs(res2))
                Gradient_loss_IF = L1Loss(
                    kornia.filters.SpatialGradient()(data_IR),
                    kornia.filters.SpatialGradient()(data_fus)
                )
                Gradient_loss_VF = L1Loss(
                    kornia.filters.SpatialGradient()(data_VIS),
                    kornia.filters.SpatialGradient()(data_fus)
                )
                Lgra = Gradient_loss_IF + Gradient_loss_VF
                # Total loss
                val_loss = 2*Lrecon + 4*L1_loss + 4*Lgra + Lsp
                loss_val.append(val_loss)
            val_epoch_loss = torch.mean(torch.tensor(loss_val))
            print('val total loss: {:.6f}'.format(val_epoch_loss))
        if val_epoch_loss < best_loss:
            best_epoch = iteration + 1
            best_loss = val_epoch_loss
            best_weights = copy.deepcopy(Net.state_dict())
            torch.save({'weight': best_weights, 'epoch': epochs},
                       os.path.join(train_path, 'best_weight.pkl'))
        scheduler1.step()
        lr_list1.append(optimizer1.state_dict()['param_groups'][0]['lr'])
        if (iteration + 1) % 50 == 0:
            weights = copy.deepcopy(Net.state_dict())
            torch.save({'weight': Net.state_dict(),
                        'epoch': iteration + 1,
                        'loss': train_epoch_loss,
                        'optimizer_state_dict': optimizer1.state_dict()},
                       os.path.join(train_path, f'weight_{iteration+1}.pkl'))
        print('best epoch:{:.0f}'.format(best_epoch))
    print('best epoch: {}, epoch_loss: {:.6f}'.format(best_epoch, best_loss))

