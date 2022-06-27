import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import argparse
import numpy as np
import math

import cv2
import matplotlib.pyplot as plt

from model.ResNet import resnet50

from utils.emotion_recognition_util import get_faces_coordinates
from utils.emotion_recognition_util import get_data_label
from utils.emotion_recognition_util import load_pretrainedmodel
from utils.emotion_evolution_util import get_b_number


transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_train_name_list = ['lm2','skjj1','xzqks3','xzqzks1','xzqzks2',
                        'yyy1','zd1','zd2','zd3','zd4',
                        '2012sjmr','bgls1','bgls2','cqjh','lm1',
                        'bd1','bd2','bhr1','bhr2','bhr3',
                        'bhr4','bhr5','bhr6','fsx1','fsx2']
                        
data_train_end_list = [ 45, 54, 83, 108, 199,
                        32, 27, 26, 80, 191,
                        27, 70, 84, 25, 174,
                        38, 78, 34, 77, 44,
                        29, 54, 42, 34, 67]
                        
data_test_name_list = ['fsx3','fsx4','fsx5','fsx6','fsx7',
                       'fsx8','mrbt1']
data_test_end_list = [31, 42, 40, 66, 55, 89, 137] 

data_train_len = len(data_train_name_list)
data_test_len = len(data_test_name_list)

path = '/data3/wangxinyu/GNEEM/data/GM-r-final/'
save_path = '/data3/wangxinyu/GNEEM/Checkpoint/resnet50_lr0001wd00005y05_l4_6/GERM_best_'

#=========init===========
device_num = 1
device = torch.device(device_num)

def lambert(x):
	exp = 1e-5
	if(x<0):
		print('Input is less than 0 !')
		return
	if (x==0):
		return 0
	y = x 
	x = np.log(x)
	while(1):
		z = np.log(y)+y
		tmp = z-x
		if(np.abs(tmp)<exp):#解的精度为0.00001
			break
		if(tmp<0):
			y = y*1.02
		if(tmp>0):
			y = y*0.98
	#y = format(y,'.4f')#保留小数后4位
	return float(y)

def E(b, rt):
    if b == 0 :
        return 1
    if b == 1:
        return 5
    return 1 + 4 * ( 1 / ( lambert( 1 / ( 1 / ( ( 1/b - 1 ) * math.exp(1/b - 1) ) + rt * math.exp(1) ) ) + 1 ) )

rental = 0.02015273341658071

def Train(model, epoch, args):
    #=========Model==========
    model.train()
    model.eval()
    # model.layer3.train()
    model.layer4.train()
    model.fc.train()
    print('==========Train Model Successfully!==========')
    # print(model)

    #=========Loss&&Optim==========
    criterion = nn.MSELoss()
    criterion_bce = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weightDecay)
    total_loss = 0.
    total_num = 0
    loss_mae = 0.
    loss_mae_num = 0
    for i in range(0, data_train_len):
        now_end = data_train_end_list[i]
        running_loss = 0.0
        total_num += now_end
        for j in range(0, now_end):
            loss_mae_num += 1
            image_path = path + data_train_name_list[i] + '/' + data_train_name_list[i] + '_' +str(j) + '.jpg'
            label_path = path + data_train_name_list[i] + '/' + data_train_name_list[i] + '_' +str(j+1) + '.txt'
            #=========Image==========
            bgr_image = cv2.imread(image_path)
            height = bgr_image.shape[0]
            weight = bgr_image.shape[1]
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

            #========Coordinate======
            face_coordinates = get_faces_coordinates(label_path)

            #========Label======
            num_arr = get_b_number(label_path)
            bi = (num_arr[1] / num_arr[0]) * 4 + 1
            bi_tensor = (torch.tensor(bi))
            bi_tensor = (bi_tensor.unsqueeze(0)).unsqueeze(1)
            bi_tensor = bi_tensor.cuda(device_num)

            labels = get_data_label(label_path)
            labels_tensor = ( torch.FloatTensor(labels) ).cuda(device_num) 

            #=========face===========
            out_labels = []
            for k, face_coor in enumerate(face_coordinates, 0):
                x1 = int( (face_coor[0] - face_coor[2]/2) * weight )
                y1 = int( (face_coor[1] - face_coor[3]/2) * height )
                x2 = int( (face_coor[0] + face_coor[2]/2) * weight )
                y2 = int( (face_coor[1] + face_coor[3]/2) * height )
                
                gray_face = gray_image[y1:y2, x1:x2]
                gray_face = cv2.resize(gray_face, (args.cropSize, args.cropSize))
                rgb_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2RGB)
                rgb_face = np.expand_dims(rgb_face, 0)
                rgb_face = np.expand_dims(rgb_face, 1)
                
                tensor_rgb_face = torch.from_numpy(rgb_face)
                tensor_rgb_face = tensor_rgb_face.squeeze(0)
                tensor_rgb_face = tensor_rgb_face.view(1, 3, args.cropSize, args.cropSize)
                tensor_rgb_face = tensor_rgb_face.float()
                tensor_rgb_face = tensor_rgb_face.to(device)

                #=========zero_grad===========
                optimizer.zero_grad()

                emotion = model(tensor_rgb_face)
                out_label = emotion
                out_labels.append(out_label.cpu().detach().numpy().item())
                loss_bce = criterion_bce(emotion, (labels_tensor[k].unsqueeze(0)).unsqueeze(0))
                loss_bce.backward()
                optimizer.step()
            
            yi = 0
            for k, p in enumerate(out_labels, 0):
                if(p > 0.5):
                    yi += 1
            yi /= len(out_labels)
            E_yi = E(yi, rental)
            E_yi_tensor = torch.tensor(E_yi)
            E_yi_tensor = (E_yi_tensor.unsqueeze(0)).unsqueeze(1)
            E_yi_tensor = E_yi_tensor.cuda(device_num)

            # print(E_yi_tensor)
            # print(bi_tensor)
            #=========forward+backward+optimize===========
            loss = criterion(E_yi_tensor, bi_tensor)
            loss_mae += abs(E_yi_tensor - bi_tensor)

            running_loss = loss.item()
            total_loss += loss.item()

            #=========print===========
            if j%args.printFreq ==  0:
                print('[ Train epoch %3d ] %s -- %3d ============== loss: %.5f' %
                    (epoch, data_train_name_list[i], j, running_loss ) )
                running_loss = 0.0

    total_loss = total_loss / total_num
    loss_mae /= loss_mae_num
    return (total_loss, loss_mae)


def Test(model, epoch, args):
    #=========Model==========
    model.eval()
    print('==========Test Model Successfully!==========')

    #=========Loss==========
    criterion = nn.MSELoss()

    total_loss = 0.
    total_num = 0
    loss_mae = 0.
    loss_mae_num = 0

    with torch.no_grad():
        for i in range(0, data_test_len):
            now_end = data_test_end_list[i]
            running_loss = 0.0
            total_num += now_end
            for j in range(0,now_end):
                loss_mae_num += 1
                image_path = path + data_test_name_list[i] + '/' + data_test_name_list[i] + '_' +str(j) + '.jpg'
                label_path = path + data_test_name_list[i] + '/' + data_test_name_list[i] + '_' +str(j+1) + '.txt'
                #=========Image==========
                bgr_image = cv2.imread(image_path)
                height = bgr_image.shape[0]
                weight = bgr_image.shape[1]
                gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

                #========Coordinate======
                face_coordinates = get_faces_coordinates(label_path)

                #========Label======
                num_arr = get_b_number(label_path)
                bi = (num_arr[1] / num_arr[0]) * 4 + 1
                bi_tensor = (torch.tensor(bi))
                bi_tensor = (bi_tensor.unsqueeze(0)).unsqueeze(1)
                bi_tensor = bi_tensor.cuda(device_num)

                #=========face===========
                out_labels = []
                for k, face_coor in enumerate(face_coordinates,0):

                    x1 = int( (face_coor[0] - face_coor[2]/2) * weight )
                    y1 = int( (face_coor[1] - face_coor[3]/2) * height )
                    x2 = int( (face_coor[0] + face_coor[2]/2) * weight )
                    y2 = int( (face_coor[1] + face_coor[3]/2) * height )
                    
                    gray_face = gray_image[y1:y2, x1:x2]
                    gray_face = cv2.resize(gray_face, (args.cropSize, args.cropSize))
                    rgb_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2RGB)
                    rgb_face = np.expand_dims(rgb_face, 0)
                    rgb_face = np.expand_dims(rgb_face, 1)
                    
                    tensor_rgb_face = torch.from_numpy(rgb_face)
                    tensor_rgb_face = tensor_rgb_face.squeeze(0)
                    tensor_rgb_face = tensor_rgb_face.view(1, 3, args.cropSize, args.cropSize)
                    tensor_rgb_face = tensor_rgb_face.float()
                    tensor_rgb_face = tensor_rgb_face.to(device)

                    emotion = model(tensor_rgb_face)
                    out_label = emotion
                    out_labels.append(out_label.cpu().detach().numpy().item())

                #=========print===========
                yi = 0.
                for k, p in enumerate(out_labels, 0):
                    if(p > 0.5):
                        yi += 1
                yi /= len(out_labels)
                E_yi = E(yi, rental)
                E_yi_tensor = torch.tensor(E_yi)
                E_yi_tensor = (E_yi_tensor.unsqueeze(0)).unsqueeze(1)
                E_yi_tensor = E_yi_tensor.cuda(device_num)

                loss = criterion(E_yi_tensor, bi_tensor) 
                loss_mae += abs(E_yi_tensor - bi_tensor)
                running_loss = loss.item()
                total_loss += loss.item()

                if j%args.printFreq ==  0:
                    print('[ Test epoch %3d ] %s -- %3d ============== Test loss: %.5f' %
                        (epoch, data_test_name_list[i], j, running_loss ) )

    total_loss = total_loss / total_num
    loss_mae /= loss_mae_num
    return (total_loss, loss_mae)


def arg_parse():
    parser = argparse.ArgumentParser(description='ResNet50 use GNMD Group Emotion Recognition Training')

    parser.add_argument('--classNum', type=int, default=1, help='numbers of classes')
    parser.add_argument('--printFreq', type=int, default=10, help='number of print frequency')

    parser.add_argument('--cropSize', type=int, default=578, help='size of crop image')

    parser.add_argument('--pretrainedModel', type=str, default='/data3/wangxinyu/pretrain_model/resnet50.pth', help='path to pretrained model')

    parser.add_argument('--epoch', type=int, default=150, help='number of total epoches to run')

    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weightDecay', type=float, default=0.0005, help='weight decay')
    
    args = parser.parse_args()
    
    return args

def print_args(args):
    print("===============================")
    print("==========  CONFIG  ===========")
    print("===============================")

    for arg, content in args.__dict__.items():
        print("{}: {}".format(arg, content))

    print("===============================")
    print("==========    END   ===========")
    print("===============================")

if __name__=="__main__":

    args = arg_parse()

    model = resnet50()
    model = load_pretrainedmodel(model,args)

    model.to(device)

    for epoch in range(0, args.epoch):
        ta = Train(model, epoch, args)

        ress = Test(model, epoch, args)

        print("Train mse loss = %.5f --------------- Test mse loss = %.5f" % (ta[0], ress[0]))
        print("Train mae loss = %.5f --------------- Test mae loss = %.5f" % (ta[1], ress[1]))
        
    