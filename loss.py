# -*- coding: utf-8 -*- 
from cv2 import sqrt
import numpy as np
import math
import matplotlib.pyplot as plt
 
from utils.emotion_evolution_util import get_b_number

b_path = '/data3/wangxinyu/GNEEM/data/GM-r-final/'
data_name_list = ['lm2','skjj1','xzqks3','xzqzks1','xzqzks2',
                  'yyy1','zd1','zd2','zd3','zd4',
                  '2012sjmr','bgls1','bgls2','cqjh','lm1',
                  'bd1','bd2','bhr1','bhr2','bhr3',
                  'bhr4','bhr5','bhr6','fsx1','fsx2',
                  'fsx3','fsx4','fsx5','fsx6','fsx7',
                  'fsx8','mrbt1']

data_end_list = [ 45, 54, 83, 108, 199,
                  32, 27, 26, 80, 191,
                  27, 70, 84, 25, 174,
                  38, 78, 34, 77, 44,
                  29, 54, 42, 34, 67,
                  31, 42, 40, 66, 55,
                  89, 137]           

x1 = []
y1 = []
x2 = []
y2 = []

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

def b(x, b0, rt):
	bb = 1 / ( lambert( 1 / ( rt * math.exp(1) * x + b0 ) )  + 1 )
	return bb

rental = 0.02015273341658071
b0 = 0.2
b0 = b0 / ( (1-b0) * math.exp( ( 1-b0 ) / b0 ) )

mae = 0.0
mse = 0.0
num = 0

for i in range(0, 32):
	now_end = data_end_list[i]
	pre = 0.0
	for j in range(0, now_end+1):
		now_b_path = b_path + data_name_list[i] + '/' + data_name_list[i] + '_' + str(j) + '.txt'
		num_arr = get_b_number(now_b_path)
		bi = num_arr[1] / num_arr[0]
		if ((j==0)or(bi!=pre)):
			y_1 = b(j, b0, rental)
			mae += abs(y_1 - bi)
			mse += (y_1 - bi)*(y_1 - bi)
			num += 1 
		pre = bi

mae /= num
mse /= num

print(mae)
print(mse)