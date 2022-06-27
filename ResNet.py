import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import argparse
import numpy as np

import cv2
import matplotlib.pyplot as plt

from model.ResNet import resnet50

from utils.emotion_recognition_util import get_faces_coordinates
from utils.emotion_recognition_util import get_data_label
from utils.emotion_recognition_util import load_pretrainedmodel


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
device = torch.device(6)


def Train(model, epoch, args):
    #=========Model==========
    model.train()
    model.eval()
    # model.layer3.train()
    model.layer4.train()
    model.fc.train()
    print('==========Train Model Successfully!==========')
    # print(model)

    #=========Count==========
    PP = 0
    NN = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    #=========Loss&&Optim==========
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weightDecay)

    for i in range(0, data_train_len):
        now_end = data_train_end_list[i]
        running_loss = 0.0
        for j in range(0, now_end+1):
            image_path = path + data_train_name_list[i] + '/' + data_train_name_list[i] + '_' +str(j) + '.jpg'
            label_path = path + data_train_name_list[i] + '/' + data_train_name_list[i] + '_' +str(j) + '.txt'
            #=========Image==========
            bgr_image = cv2.imread(image_path)
            height = bgr_image.shape[0]
            weight = bgr_image.shape[1]
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

            #========Coordinate======
            face_coordinates = get_faces_coordinates(label_path)

            #========Label======
            labels = get_data_label(label_path)
            labels_tensor = ( torch.FloatTensor(labels) ).cuda(6) 

            #=========face===========
            out_labels = []
            for k, face_coor in enumerate(face_coordinates, 0):
                if labels[k] == 1 and PP > NN:
                    continue
                if labels[k] == 1:
                    PP += 1
                else:
                    NN += 1
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

                #=========forward+backward+optimize===========
                emotion = model(tensor_rgb_face)
                loss = criterion(emotion, (labels_tensor[k].unsqueeze(0)).unsqueeze(0))
                out_label = emotion
                out_labels.append(out_label.cpu().detach().numpy().item())
                loss.backward()
                optimizer.step()

            #=========print===========
            running_loss = loss.item()
            if j%args.printFreq ==  0:
                print('[ Train epoch %3d ] %s -- %3d ============== loss: %.5f' %
                    (epoch, data_train_name_list[i], j, running_loss ) )
                running_loss = 0.0

            for k, p in enumerate(out_labels, 0):
                if(p > 0.5):
                    if(labels[k] == 1): TP += 1
                    else: FP += 1
                else:
                    if(labels[k] == 1): FN += 1
                    else: TN += 1

    if((TP+FP+TN+FN)!=0): Accuracy = (TP + TN) / (TP + FP + TN + FN)
    print('Train_Accuracy ============= %.10f' % Accuracy)

    print('=========Finished Training===========')

    return Accuracy
    
                 
def Test(model, epoch, args):
    #=========Model==========
    model.eval()
    print('==========Test Model Successfully!==========')

    #=========Loss==========
    criterion = nn.BCELoss()

    TP = 0
    FP = 0
    FN = 0
    TN = 0 

    total_loss = 0.

    with torch.no_grad():
        for i in range(0, data_test_len):
            now_end = data_test_end_list[i]
            running_loss = 0.0
            now_loss = 0.
            for j in range(0,now_end+1):
                image_path = path + data_test_name_list[i] + '/' + data_test_name_list[i] + '_' +str(j) + '.jpg'
                label_path = path + data_test_name_list[i] + '/' + data_test_name_list[i] + '_' +str(j) + '.txt'
                #=========Image==========
                bgr_image = cv2.imread(image_path)
                height = bgr_image.shape[0]
                weight = bgr_image.shape[1]
                gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

                #========Coordinate======
                face_coordinates = get_faces_coordinates(label_path)

                #========Label======
                labels = get_data_label(label_path)
                labels_tensor = ( torch.FloatTensor(labels) ).cuda(6) 

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
                    
                    #=========loss===========
                    loss = criterion(emotion, (labels_tensor[k].unsqueeze(0)).unsqueeze(0) )
                    running_loss += loss.item()
                    out_label = emotion
                    # print(emotion)
                    out_labels.append(out_label.cpu().detach().numpy().item())
                #=========print===========
                running_loss = running_loss / len(face_coordinates)
                now_loss += running_loss
                if j%args.printFreq ==  0:
                    print('[ Test epoch %3d ] %s -- %3d ============== Test loss: %.5f' %
                        (epoch, data_test_name_list[i], j, running_loss ) )
                running_loss = 0.0
                for k, p in enumerate(out_labels, 0):
                    if(p > 0.5):
                        if(labels[k] == 1): TP += 1
                        else: FP += 1
                    else:
                        if(labels[k] == 1): FN += 1
                        else: TN += 1
            now_loss = now_loss / now_end
            total_loss += now_loss
        total_loss = total_loss / data_test_len
        Accuracy = (TP + TN) / (TP + FP + TN + FN)
        print('Accuracy ============= %.5f' % Accuracy)

    return (Accuracy,total_loss)


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

    bestAcc = 0
    bestepoch = 0

    x_acc = []
    y_acc = []
    x_tacc = []
    y_tacc = []
    x_loss = []
    y_loss = []

    colors1 = '#CC0000' #点的颜色
    colors2 = '#00CC00'
    colorsb = '#0000CC'

    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(xmax=150,xmin=0)
    plt.ylim(ymax=1,ymin=0)
    
    plt.plot(x_acc, y_acc, linewidth=1, c=colors1, alpha=0.4, label='Test_Accuracy')
    plt.plot(x_acc, y_acc, linewidth=1, c=colors2, alpha=0.4, label='Train_Accuracy')
    plt.legend()
    
    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(xmax=150,xmin=0)
    plt.ylim(ymax=20,ymin=0)

    for epoch in range(0, args.epoch):
        ta = Train(model, epoch, args)
        x_tacc.append(epoch)
        y_tacc.append(ta)

        ress = Test(model, epoch, args)
        y_loss.append(ress[1])
        x_loss.append(epoch)
        ans = ress[0]
        
        if(ans > bestAcc):
            bestAcc = ans
            bestepoch = epoch
            best_path = save_path + str(epoch) +'.pth'
            torch.save(model.state_dict(), best_path)
        print('now accurary %.15f ---- best accurary %.15f ---- now loss %.15f' % (ans, bestAcc, ress[1]))

        x_acc.append(epoch)
        y_acc.append(ans)
        # 画折线图
        plt.figure(1)
        plt.plot(x_acc, y_acc, linewidth=1, c=colors1, alpha=0.4, label='Test_Accuracy')
        plt.plot(x_tacc, y_tacc, linewidth=1, c=colors2, alpha=0.4, label='Train_Accuracy')
        plt.savefig('/data3/wangxinyu/GNEEM/Checkpoint/resnet50_lr0001wd00005y05_l4_6/result_acc.jpg', dpi=300)
        plt.show()

        plt.figure(2)
        plt.plot(x_loss, y_loss, linewidth=1, c=colorsb, alpha=0.4, label='Test_Loss')
        plt.savefig('/data3/wangxinyu/GNEEM/Checkpoint/resnet50_lr0001wd00005y05_l4_6/result_loss.jpg', dpi=300)
        plt.show()

    print('Best Accuracy ============= %.10f' % bestAcc)

    
                


