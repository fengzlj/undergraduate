import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

import numpy as np
import cv2

import matplotlib.pyplot as plt

from model.model import Mini_Xception
from utils.emotion_recognition_util import get_faces_coordinates
from utils.emotion_recognition_util import get_data_label


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
save_path = '/data3/wangxinyu/GNEEM/checkpoint/lr0001nwdy05/GERM_best_'

#=========init===========
device = torch.device(7)

def Train(model, epoch):
    #=========Model==========
    model.train()
    print('==========Train Model Successfully!==========')
    # print(model)

    PP = 0
    NN = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    #=========Loss&&Optim==========
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0)

    for i in range(0, data_train_len):
        # if(data_train_name_list[i] == 'xzqzks1' or data_train_name_list[i] == '2012sjmr'): continue
        now_end = data_train_end_list[i]
        running_loss = 0.0
        for j in range(0, now_end+1):
            image_path = path + data_train_name_list[i] + '/' + data_train_name_list[i] + '_' +str(j) + '.jpg'
            label_path = path + data_train_name_list[i] + '/' + data_train_name_list[i] + '_' +str(j) + '.txt'
            #=========Image==========
            bgr_image = cv2.imread(image_path)
            height = bgr_image.shape[0]
            weight = bgr_image.shape[1]
            # print(height,' ',weight)
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

            #========Coordinate======
            face_coordinates = get_faces_coordinates(label_path)

            #========Label======
            labels = get_data_label(label_path)
            labels_tensor = ( torch.FloatTensor(labels) ).cuda(7) 

            #=========face===========
            out_labels = []
            for k, face_coor in enumerate(face_coordinates,0):
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
                # print(face_coor[0],' ',face_coor[1],' ',face_coor[2],' ',face_coor[3])
                # print(x1,' ',y1,' ',x2,' ',y2)
                
                gray_face = gray_image[y1:y2, x1:x2]
                gray_face = cv2.resize(gray_face, (48, 48))
                # gray_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2RGB)
                # cv2.imwrite(save_path, gray_face)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, 1)
                
                tensor_gray_face = torch.from_numpy(gray_face)
                tensor_gray_face = tensor_gray_face.float()
                tensor_gray_face = tensor_gray_face.to(device)
                # print(type(tensor_gray_face),' ',tensor_gray_face.size())

                #=========zero_grad===========
                optimizer.zero_grad()

                #=========forward+backward+optimize===========
                emotion = model(tensor_gray_face)
                loss = criterion(emotion, (((labels_tensor[k].unsqueeze(0)).unsqueeze(0)).unsqueeze(0)).unsqueeze(0))
                out_label = emotion
                out_labels.append(out_label.cpu().detach().numpy().item())
                loss.backward()
                optimizer.step()

            #=========print===========
            running_loss += loss.item()
            if j%10 ==  0:
                print('[Train epoch %2d ] %s -- %5d ============== loss: %.3f' %
                    (epoch, data_train_name_list[i], j, running_loss/10 ) )
                running_loss = 0.0

            for k, p in enumerate(out_labels, 0):
                if(p > 0.5):
                    if(labels[k] == 1): TP += 1
                    else: FP += 1
                else:
                    if(labels[k] == 1): FN += 1
                    else: TN += 1
    Recall = 0.0
    Precision = 0.0
    Accuracy = 0.0
    if((TP+FN)!=0): Recall = TP / (TP + FN)  
    if((TP+FP)!=0): Precision = TP /(TP + FP)  
    if((TP+FP+TN+FN)!=0): Accuracy = (TP + TN) / (TP + FP + TN + FN)
    print('Train_Recall ============= %.5f' % Recall)
    print('Train_Precision ============= %.5f' % Precision)
    print('Train_Accuracy ============= %.5f' % Accuracy)

    print('=========Finished Training===========')

    return Accuracy
                 
def Test(model, epoch):
    #=========Model==========
    # checkpoint = torch.load(save_path, map_location=device)
    # model = Mini_Xception()
    # model.load_state_dict(checkpoint, strict=False)
    # model.to(device)
    model.eval()
    print('==========Test Model Successfully!==========')
    # print(model)

    #=========Loss==========
    criterion = nn.BCELoss()

    TP = 0
    FP = 0
    FN = 0
    TN = 0 

    for i in range(0, data_test_len):
        now_end = data_test_end_list[i]
        running_loss = 0.0
        for j in range(0,now_end+1):
            image_path = path + data_test_name_list[i] + '/' + data_test_name_list[i] + '_' +str(j) + '.jpg'
            label_path = path + data_test_name_list[i] + '/' + data_test_name_list[i] + '_' +str(j) + '.txt'
            #=========Image==========
            bgr_image = cv2.imread(image_path)
            height = bgr_image.shape[0]
            weight = bgr_image.shape[1]
            # print(height,' ',weight)
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

            #========Coordinate======
            face_coordinates = get_faces_coordinates(label_path)

            #========Label======
            labels = get_data_label(label_path)
            labels_tensor = ( torch.FloatTensor(labels) ).cuda(7) 

            #=========face===========
            out_labels = []
            for k, face_coor in enumerate(face_coordinates,0):
                x1 = int( (face_coor[0] - face_coor[2]/2) * weight )
                y1 = int( (face_coor[1] - face_coor[3]/2) * height )
                x2 = int( (face_coor[0] + face_coor[2]/2) * weight )
                y2 = int( (face_coor[1] + face_coor[3]/2) * height )
                # print(face_coor[0],' ',face_coor[1],' ',face_coor[2],' ',face_coor[3])
                # print(x1,' ',y1,' ',x2,' ',y2)
                
                gray_face = gray_image[y1:y2, x1:x2]
                gray_face = cv2.resize(gray_face, (48, 48))
                # gray_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2RGB)
                # cv2.imwrite(save_path, gray_face)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, 1)
                
                tensor_gray_face = torch.from_numpy(gray_face)
                tensor_gray_face = tensor_gray_face.float()
                tensor_gray_face = tensor_gray_face.to(device)
                # print(type(tensor_gray_face),' ',tensor_gray_face.size())

                emotion = model(tensor_gray_face)
                
                #=========loss===========
                loss = criterion(emotion, (((labels_tensor[k].unsqueeze(0)).unsqueeze(0)).unsqueeze(0)).unsqueeze(0))
                out_label = emotion
                out_labels.append(out_label.cpu().detach().numpy().item())
            #=========print===========
            running_loss += loss.item()
            if j%10 ==  0:
                print('[Test epoch %2d ] %s -- %5d ============== Test loss: %.3f' %
                    (epoch, data_test_name_list[i], j, running_loss/10 ) )
                running_loss = 0.0
            for k, p in enumerate(out_labels, 0):
                if(p > 0.4):
                    if(labels[k] == 1): TP += 1
                    else: FP += 1
                else:
                    if(labels[k] == 1): FN += 1
                    else: TN += 1
    Recall = 0.0
    Precision = 0.0
    Accuracy = 0.0
    if((TP+FN)!=0): Recall = TP / (TP + FN)  
    if((TP+FP)!=0): Precision = TP /(TP + FP)  
    if((TP+FP+TN+FN)!=0): Accuracy = (TP + TN) / (TP + FP + TN + FN)
    print('Recall ============= %.5f' % Recall)
    print('Precision ============= %.5f' % Precision)
    print('Accuracy ============= %.5f' % Accuracy)

    return (Recall, Precision, Accuracy)

if __name__=="__main__":

    model = Mini_Xception()
    model.to(device)
    bestAcc = 0
    bestepoch = 0
    bestRec = 0 
    bestPre = 0

    x_rec = []
    y_rec = []
    x_pre = []
    y_pre = []
    x_acc = []
    y_acc = []
    x_tacc = []
    y_tacc = []
    

    colors1 = '#CC0000' #点的颜色
    colors2 = '#00CC00'
    colors3 = '#0000CC' 
    colors4 = '#666666' 

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax=200,xmin=0)
    plt.ylim(ymax=1,ymin=0)

    plt.plot(x_rec, y_rec, linewidth=1, c=colors1, alpha=0.4, label='Recall')
    plt.plot(x_pre, y_pre, linewidth=1, c=colors2, alpha=0.4, label='Precision')
    plt.plot(x_acc, y_acc, linewidth=1, c=colors3, alpha=0.4, label='Accuracy')
    plt.plot(x_acc, y_acc, linewidth=1, c=colors4, alpha=0.4, label='T_Accuracy')
    plt.legend()
    
    for epoch in range(0, 100):
        ta = Train(model, epoch)
        x_tacc.append(epoch)
        y_tacc.append(ta)
        ans = Test(model, epoch)
        print(ans[2],' ---- ',bestAcc)
        if(ans[2] > bestAcc):
            bestAcc = ans[2]
            bestepoch = epoch
            bestRec = ans[0]
            bestPre = ans[1]
            best_path = save_path + str(epoch) +'.pth'
            torch.save(model.state_dict(), best_path)

        x_rec.append(epoch)
        y_rec.append(ans[0])
        x_pre.append(epoch)
        y_pre.append(ans[1])
        x_acc.append(epoch)
        y_acc.append(ans[2])
        # 画折线图
        plt.plot(x_rec, y_rec, linewidth=1, c=colors1, alpha=0.4, label='Recall')
        plt.plot(x_pre, y_pre, linewidth=1, c=colors2, alpha=0.4, label='Precision')
        plt.plot(x_acc, y_acc, linewidth=1, c=colors3, alpha=0.4, label='Accuracy')
        plt.plot(x_tacc, y_tacc, linewidth=1, c=colors4, alpha=0.4, label='T_Accuracy')
        plt.savefig('/data3/wangxinyu/GNEEM/checkpoint/lr0001nwdy05/recpreacc.jpg', dpi=300)
        plt.show()

    print('Best Accuracy ============= %.5f' % bestAcc)

    
                


