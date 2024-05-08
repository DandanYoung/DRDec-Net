import os
import torch
from PIL import Image
from model import ImgFusNet
import numpy as np
import argparse
import cv2


def saveimage(x, savepath):
    x = x.cpu().numpy()[0, :, :, :]
    x = np.transpose(x, (1, 2, 0))
    cv2.imwrite(savepath, x * 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device('cuda:0'))
    opt = parser.parse_args()
    # =============================================================================
    # Test Details
    # =============================================================================
    test_data_path = './datasets/data/test'
    # Determine the number of
    # files
    Test_Image_Number = len(os.listdir(test_data_path+'/test_ir'))
    # =============================================================================
    # Test
    # =============================================================================
    for i in range(int(Test_Image_Number)):
        Test_IR = Image.open(test_data_path + '/test_ir/' + str(i + 1) + '.png')  # infrared image
        Test_Vis = Image.open(test_data_path + '/test_vi/' + str(i + 1) + '.png')  # visible image

        Net = ImgFusNet().to(opt.device)
        Net.load_state_dict(torch.load("./output/best_weight.pkl")['weight'])

        Net.eval()
        img_test1 = np.array(Test_IR, dtype='float32') / 255  # 将其转换为一个矩阵
        img_test1 = torch.from_numpy(img_test1.reshape((1, 1, img_test1.shape[0], img_test1.shape[1])))

        img_test2 = np.array(Test_Vis, dtype='float32') / 255  # 将其转换为一个矩阵
        img_test2 = torch.from_numpy(img_test2.reshape((1, 1, img_test2.shape[0], img_test2.shape[1])))

        img_test1 = img_test1.to(opt.device)
        img_test2 = img_test2.to(opt.device)
        print('正在测试第%d对' % (i + 1))

        with torch.no_grad():
            data_fus, _, _, _, _ = Net(img_test1, img_test2, 3)
        saveimage(data_fus, './result/' + str(i + 1) + '.png')