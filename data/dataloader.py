data_name_list = ['fsx3','fsx4','fsx5','fsx6','fsx7',
                  'fsx8','mrbt1']
                #   ['lm2','skjj1','xzqks3','xzqzks1','xzqzks2',
                #   'yyy1','zd1','zd2','zd3','zd4',
                #   '2012sjmr','bgls1','bgls2','cqjh','lm1',
                #   'bd1','bd2','bhr1','bhr2','bhr3',
                #   'bhr4','bhr5','bhr6','fsx1','fsx2']
                  

data_end_list = [31, 42, 40, 66, 55,
                  89, 137] 
                #   [ 45, 54, 83, 108, 199,
                #   32, 27, 26, 80, 191,
                #   27, 70, 84, 25, 174,
                #   38, 78, 34, 77, 44,
                #   29, 54, 42, 34, 67]
                                               


# sum              
a = 0
num = 0
for i in data_end_list:
    a += i
    num += 1

print(num)
print(a)