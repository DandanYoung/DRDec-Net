import torch
import torch.nn as nn


class SpaAttBlock(nn.Module):
    def __init__(self):
        super(SpaAttBlock, self).__init__()
        self.fc1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2, 32, (3, 3), 1, padding=0),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, (3, 3), 1, padding=0)
        )
        self.fc2 = nn.Sigmoid()

    def forward(self, res1, res2):
        block1 = torch.cat([res1, res2], dim=1)
        block2 = self.fc1(block1)
        block3 = self.fc2(block2)
        return block3


class ResFusion(nn.Module):
    def __init__(self):
        super(ResFusion, self).__init__()
        self.SpaAttBlock = SpaAttBlock()
        self.fc1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 32, (3, 3), 1, padding=0)
        )
        self.fc2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 32, (3, 3), 1, padding=0)
        )
        self.fc3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, (3, 3), 1, padding=0),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, (3, 3), 1, padding=0),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 1, (3, 3), 1, padding=0)
        )

    def forward(self, res1, res2):
        block1 = self.SpaAttBlock(res1, res2)
        block2 = self.fc1(res1)
        block3 = self.fc2(res2)
        block2 = torch.mul(block2, block1)
        block3 = torch.mul(block3, block1)
        block4 = torch.cat([block2, block3], dim=1)
        res_f = self.fc3(block4)
        return res_f


class RoDec(nn.Module):
    def __init__(self):
        super(RoDec, self).__init__()
        self.fc1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2, 32, 3, 1, padding=0),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, 3, 1, padding=0),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Conv2d(64, 64, 1, 1, 0)
        self.fc4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.Sigmoid()
        )
        self.fc5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.Sigmoid()
        )
        self.fc6 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 1, 3, 1, padding=0)
        )
        self.fc7 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 1, 3, 1, padding=0)
        )

    def forward(self, img):
        block1 = self.fc1(img)
        block2 = block1 + self.fc2(block1)
        block3 = torch.cat([block1, block2], dim=1)
        block4 = self.fc3(block3)
        block5 = torch.mul(block4, self.fc4(block4))
        block6 = torch.mul(block4, self.fc5(block4))
        block7 = self.fc6(block5)
        block8 = self.fc7(block6)
        ro_left = torch.mean(block7, dim=3, keepdims=True)
        ro_right = torch.mean(block8, dim=2, keepdims=True)
        ro_c = torch.matmul(ro_left, ro_right)
        return ro_c


class RoComFusion(nn.Module):
    def __init__(self):
        super(RoComFusion, self).__init__()
        self.fc1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, (3, 3), 1, padding=0),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, (3, 3), 1, padding=0),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, (3, 3), 1, padding=0),
            nn.LeakyReLU()
        )
        self.fc4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 16, (3, 3), 1, padding=0),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 1, (3, 3), 1, padding=0),
            # nn.Conv2d(in_channels=,out_channels=,kernel_size=,stride=,padding=)
        )

    def forward(self, ro_batch):
        block1 = self.fc1(ro_batch)
        block2 = block1 + self.fc2(block1)
        block3 = block2 + self.fc3(block2)
        ro_cf = self.fc4(block3)
        return ro_cf


class ImgFusNet(nn.Module):
    def __init__(self):
        super(ImgFusNet, self).__init__()
        self.ResFusion = ResFusion()
        self.RoDec = RoDec()
        self.RoComFusion = RoComFusion()

    def forward(self, img_1, img_2, block_num):
        ro_list = []
        img1 = img_1
        img2 = img_2
        for i in range(block_num):
            img = torch.cat([img1, img2], dim=1)
            ro_c = self.RoDec(img)
            ro_list.append(ro_c)
            img1 = img1 - ro_c
            img2 = img2 - ro_c
        res_f = self.ResFusion(img1, img2)
        ro_batch = torch.cat(ro_list, dim=1)
        ro_cf = self.RoComFusion(ro_batch)
        print(res_f.shape, ro_cf.shape)
        img_f = res_f + ro_cf
        ro_sum = sum(ro_list)
        img_i = ro_sum + img1
        img_v = ro_sum + img2
        return img_f, img_i, img_v, img1, img2


if __name__ == "__main__":
    ImgFusNet = ImgFusNet()
    a = torch.FloatTensor(size=(2, 1, 120, 120))
    b = torch.FloatTensor(size=(2, 1, 120, 120))
    output_fused, _, _, _, _ = ImgFusNet(a, b, 3)
    print(output_fused.shape)