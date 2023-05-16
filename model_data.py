import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from quantize import*

parser = argparse.ArgumentParser(description = 'setting')
parser.add_argument('--w_bit', type = int, default = 1)
parser.add_argument('--a_bit', type = int, default = 3)
args = parser.parse_args(args = [])

# PATH_pm = "./mnist_adam_round(2^k)_1_3_decimal0.pt"
# PATH_cm1 = "./mnist_adam_round(2^k-1)_1_3_decimal0.pt"
# PATH_cm2 = "./mnist_floor(2^k-1)_1_3.pt"
# PATH_cm3 = "./mnist_floor(2^k)_1_3.pt"

PATH_1 = "./mnist_adam_floor(2^k-1)_1_3.pt"
PATH_2 = "./mnist_adam_floor(2^k)_1_3.pt"

PATH_3 = "./mnist_adam_round(2^k-1)_1_3.pt"
PATH_4 = "./mnist_adam_round(2^k)_1_3.pt"

PATH_5 = "./mnist_adam_mix(2^k-1)_1_3.pt"
PATH_6 = "./mnist_adam_mix(2^k)_1_3.pt"

# PATH_pm = "./mnist_round(2^k)_4_3.pt"
# PATH_cm1 = "./mnist_round(2^k-1)_4_3.pt"
# PATH_cm2 = "./mnist_floor(2^k)_4_3.pt"
# PATH_cm3 = "./mnist_floor(2^k-1)_4_3.pt"

# PATH_pm = "./mnist_round(2^k)_3_3.pt"
# PATH_cm1 = "./mnist_round(2^k-1)_3_3.pt"
# PATH_cm2 = "./mnist_floor(2^k)_3_3.pt"
# PATH_cm3 = "./mnist_floor(2^k-1)_3_3.pt"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        BinaryConv2d = conv2d_Q_fn(args.w_bit)
        BinaryLinear = linear_Q_fn(args.w_bit)
        self.act = activation_quantize_fn(args.a_bit)
        self.conv1 = BinaryConv2d(1, 1, 5, bias = False)
        self.conv2 = BinaryConv2d(1, 4, 5, bias = False)
        self.fc1 = BinaryLinear(4*4*4, 16, bias = True)
        self.fc2 = BinaryLinear(16, 10, bias = True)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.conv1(x) #畳み込み第1層
        x = F.max_pool2d(x,(2,2)) #プーリング第1層 フィルタサイズ2*2
        x = torch.clamp(x, 0, 0.875) #(入力x,最小値,最大値) 入力が最小値~最大値の範囲外の場合、最小値(最大値)に変換する
        x = self.act(x) #出力の量子化
        x = self.conv2(x) #畳み込み第2層
        x = F.max_pool2d(x,(2,2)) #プーリング第2層 フィルタサイズ2*2
        x = torch.clamp(x, 0, 0.875) #この場合、入力が0より小さい場合は0に、0.875より大きい場合は0.875に変換
        x = self.act(x) #出力の量子化
        x = x.view(-1, 4*4*4) #１つ目の引数に-1を入れることで、２つ目の引数で指定した値にサイズ数を自動的に調整
        x = self.fc1(x) 
        x = self.fc2(x) #MNISTは0~9までの10分類なので出力10
        x = self.soft(x) #活性化関数
        return x

model = Net()
print('\n**********ネットワーク概要**********')
print(model)

# PATH = [PATH_pm, PATH_cm1, PATH_cm2, PATH_cm3]

PATH = [PATH_1,PATH_2,PATH_3,PATH_4,PATH_5,PATH_6]

# for path in PATH:
    
#     if path == PATH_pm:
#         print('\n\n\n//round(2^k)')
#     elif path == PATH_cm1:
#         print('\n\n\n//round(2^k-1)')
#     elif path == PATH_cm2:
#         print('\n\n\n//floor(2^k)')
#     else:
#         print('\n\n\n//floor(2^k-1)')

for path in PATH:
    
    if path ==PATH_1:
        print('\n\n\n//adam_floor(2^k-1)_1_3')
    elif path == PATH_2:
        print('\n\n\n//adam_floor(2^k)_1_3')
    elif path == PATH_3:
        print('\n\n\n//adam_round(2^k-1)_1_3')
    elif path == PATH_4:
        print('\n\n\n//adam_round(2^k)_1_3')
    elif path == PATH_5:
        print('\n\n\n//adam_mix(2^k-1)_1_3')
    else:
        print('\n\n\n//adam_mix(2^k)_1_3')
    
    model.load_state_dict(torch.load(path)) # 保存したモデルパラメータの読み込み

    model_data = model.state_dict() #モデルのパラメータやオプティマイザの状態を表す
    # print('\n**********重み**********')
    # print(model_data)

#model_dataは辞書型 model_data[key]でvalue
    weight_conv1 = model_data['conv1.weight']
    weight_conv2 = model_data['conv2.weight']
    weight_fc1 = model_data['fc1.weight']
    bias_fc1 = model_data['fc1.bias']
    weight_fc2 = model_data['fc2.weight']
    bias_fc2 = model_data['fc2.bias']


    data_list = [weight_conv1, weight_conv2, weight_fc1, bias_fc1 , weight_fc2, bias_fc2]

    # print('\n**********量子化重み**********')

    for i, data in enumerate(data_list):
        data =  torch.sign(data) #負の値は-1、正の値は1、0は0が返ってくる
        data = torch.clamp(data, 0, 1) #(入力x,最小値,最大値) 入力が最小値~最大値の範囲外の場合、最小値(最大値)に変換する 
        #verilogでは0か1のみなので-1を0としている
        if i == 0: #weight_conv1 [1,1,5,5]
            print('\n//conv1.weight')
            # print(data)

            data = data.int().tolist() #dataをint型にしてさらにlistにする
            #dataは4次元、data_1は1次元
            data_1 = data[0][0][0] + data[0][0][1] + data[0][0][2] + data[0][0][3] + data[0][0][4]
            print('\nparameter filter_28_24 = 25!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{};'.format(data_1[0],data_1[1],data_1[2],data_1[3],data_1[4],data_1[5],data_1[6],data_1[7],data_1[8],data_1[9],data_1[10],data_1[11],data_1[12],data_1[13],data_1[14],data_1[15],data_1[16],data_1[17],data_1[18],data_1[19],data_1[20],data_1[21],data_1[22],data_1[23],data_1[24]))

        elif i == 1: #weight_conv2 [4,1,5,5]
            print('\n//conv2.weight')
            # print(data)

            data = data.int().tolist()

            #dataは4次元、data_1は1次元
            data_1_1 = data[0][0][0] + data[0][0][1] + data[0][0][2] + data[0][0][3] + data[0][0][4]
            data_2_1 = data[1][0][0] + data[1][0][1] + data[1][0][2] + data[1][0][3] + data[1][0][4]
            data_3_1 = data[2][0][0] + data[2][0][1] + data[2][0][2] + data[2][0][3] + data[2][0][4]
            data_4_1 = data[3][0][0] + data[3][0][1] + data[3][0][2] + data[3][0][3] + data[3][0][4]
            
            
            print('\nparameter filter_12_8_1 = 25!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{};'.format(data_1_1[0],data_1_1[1],data_1_1[2],data_1_1[3],data_1_1[4],data_1_1[5],data_1_1[6],data_1_1[7],data_1_1[8],data_1_1[9],data_1_1[10],data_1_1[11],data_1_1[12],data_1_1[13],data_1_1[14],data_1_1[15],data_1_1[16],data_1_1[17],data_1_1[18],data_1_1[19],data_1_1[20],data_1_1[21],data_1_1[22],data_1_1[23],data_1_1[24]))


            print('parameter filter_12_8_2 = 25!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{};'.format(data_2_1[0],data_2_1[1],data_2_1[2],data_2_1[3],data_2_1[4],data_2_1[5],data_2_1[6],data_2_1[7],data_2_1[8],data_2_1[9],data_2_1[10],data_2_1[11],data_2_1[12],data_2_1[13],data_2_1[14],data_2_1[15],data_2_1[16],data_2_1[17],data_2_1[18],data_2_1[19],data_2_1[20],data_2_1[21],data_2_1[22],data_2_1[23],data_2_1[24]))


            print('parameter filter_12_8_3 = 25!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{};'.format(data_3_1[0],data_3_1[1],data_3_1[2],data_3_1[3],data_3_1[4],data_3_1[5],data_3_1[6],data_3_1[7],data_3_1[8],data_3_1[9],data_3_1[10],data_3_1[11],data_3_1[12],data_3_1[13],data_3_1[14],data_3_1[15],data_3_1[16],data_3_1[17],data_3_1[18],data_3_1[19],data_3_1[20],data_3_1[21],data_3_1[22],data_3_1[23],data_3_1[24]))


            print('parameter filter_12_8_4 = 25!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{};'.format(data_4_1[0],data_4_1[1],data_4_1[2],data_4_1[3],data_4_1[4],data_4_1[5],data_4_1[6],data_4_1[7],data_4_1[8],data_4_1[9],data_4_1[10],data_4_1[11],data_4_1[12],data_4_1[13],data_4_1[14],data_4_1[15],data_4_1[16],data_4_1[17],data_4_1[18],data_4_1[19],data_4_1[20],data_4_1[21],data_4_1[22],data_4_1[23],data_4_1[24]))

        elif i == 2: #weight_fc1 [16,64]
            print('\n//fc1.weight')
            # print(data)

            data = data.int().tolist()
            # print(data)

            data_1 = data[0] #64個
            data_2 = data[1] #64個
            data_3 = data[2] #64個
            data_4 = data[3] #64個
            data_5 = data[4] #64個
            data_6 = data[5] #64個
            data_7 = data[6] #64個
            data_8 = data[7] #64個
            data_9 = data[8] #64個
            data_10 = data[9] #64個
            data_11 = data[10] #64個
            data_12 = data[11] #64個
            data_13 = data[12] #64個
            data_14 = data[13] #64個
            data_15 = data[14] #64個
            data_16 = data[15] #64個


            print('\nparameter fc_64_16_0 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_1[0],data_1[1],data_1[2],data_1[3],data_1[4],data_1[5],data_1[6],data_1[7],data_1[8],data_1[9],data_1[10],data_1[11],data_1[12],data_1[13],data_1[14],data_1[15],data_1[16],data_1[17],data_1[18],data_1[19],data_1[20],data_1[21],data_1[22],data_1[23],data_1[24],data_1[25],data_1[26],data_1[27],data_1[28],data_1[29],data_1[30],data_1[31],data_1[32],data_1[33],data_1[34],data_1[35],data_1[36],data_1[37],data_1[38],data_1[39],data_1[40],data_1[41],data_1[42],data_1[43],data_1[44],data_1[45],data_1[46],data_1[47],data_1[48],data_1[49],data_1[50],data_1[51],data_1[52],data_1[53],data_1[54],data_1[55],data_1[56],data_1[57],data_1[58],data_1[59],data_1[60],data_1[61],data_1[62],data_1[63]))

            print('parameter fc_64_16_1 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_2[0],data_2[1],data_2[2],data_2[3],data_2[4],data_2[5],data_2[6],data_2[7],data_2[8],data_2[9],data_2[10],data_2[11],data_2[12],data_2[13],data_2[14],data_2[15],data_2[16],data_2[17],data_2[18],data_2[19],data_2[20],data_2[21],data_2[22],data_2[23],data_2[24],data_2[25],data_2[26],data_2[27],data_2[28],data_2[29],data_2[30],data_2[31],data_2[32],data_2[33],data_2[34],data_2[35],data_2[36],data_2[37],data_2[38],data_2[39],data_2[40],data_2[41],data_2[42],data_2[43],data_2[44],data_2[45],data_2[46],data_2[47],data_2[48],data_2[49],data_2[50],data_2[51],data_2[52],data_2[53],data_2[54],data_2[55],data_2[56],data_2[57],data_2[58],data_2[59],data_2[60],data_2[61],data_2[62],data_2[63]))

            print('parameter fc_64_16_2 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_3[0],data_3[1],data_3[2],data_3[3],data_3[4],data_3[5],data_3[6],data_3[7],data_3[8],data_3[9],data_3[10],data_3[11],data_3[12],data_3[13],data_3[14],data_3[15],data_3[16],data_3[17],data_3[18],data_3[19],data_3[20],data_3[21],data_3[22],data_3[23],data_3[24],data_3[25],data_3[26],data_3[27],data_3[28],data_3[29],data_3[30],data_3[31],data_3[32],data_3[33],data_3[34],data_3[35],data_3[36],data_3[37],data_3[38],data_3[39],data_3[40],data_3[41],data_3[42],data_3[43],data_3[44],data_3[45],data_3[46],data_3[47],data_3[48],data_3[49],data_3[50],data_3[51],data_3[52],data_3[53],data_3[54],data_3[55],data_3[56],data_3[57],data_3[58],data_3[59],data_3[60],data_3[61],data_3[62],data_3[63]))

            print('parameter fc_64_16_3 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_4[0],data_4[1],data_4[2],data_4[3],data_4[4],data_4[5],data_4[6],data_4[7],data_4[8],data_4[9],data_4[10],data_4[11],data_4[12],data_4[13],data_4[14],data_4[15],data_4[16],data_4[17],data_4[18],data_4[19],data_4[20],data_4[21],data_4[22],data_4[23],data_4[24],data_4[25],data_4[26],data_4[27],data_4[28],data_4[29],data_4[30],data_4[31],data_4[32],data_4[33],data_4[34],data_4[35],data_4[36],data_4[37],data_4[38],data_4[39],data_4[40],data_4[41],data_4[42],data_4[43],data_4[44],data_4[45],data_4[46],data_4[47],data_4[48],data_4[49],data_4[50],data_4[51],data_4[52],data_4[53],data_4[54],data_4[55],data_4[56],data_4[57],data_4[58],data_4[59],data_4[60],data_4[61],data_4[62],data_4[63]))

            print('parameter fc_64_16_4 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_5[0],data_5[1],data_5[2],data_5[3],data_5[4],data_5[5],data_5[6],data_5[7],data_5[8],data_5[9],data_5[10],data_5[11],data_5[12],data_5[13],data_5[14],data_5[15],data_5[16],data_5[17],data_5[18],data_5[19],data_5[20],data_5[21],data_5[22],data_5[23],data_5[24],data_5[25],data_5[26],data_5[27],data_5[28],data_5[29],data_5[30],data_5[31],data_5[32],data_5[33],data_5[34],data_5[35],data_5[36],data_5[37],data_5[38],data_5[39],data_5[40],data_5[41],data_5[42],data_5[43],data_5[44],data_5[45],data_5[46],data_5[47],data_5[48],data_5[49],data_5[50],data_5[51],data_5[52],data_5[53],data_5[54],data_5[55],data_5[56],data_5[57],data_5[58],data_5[59],data_5[60],data_5[61],data_5[62],data_5[63]))

            print('parameter fc_64_16_5 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_6[0],data_6[1],data_6[2],data_6[3],data_6[4],data_6[5],data_6[6],data_6[7],data_6[8],data_6[9],data_6[10],data_6[11],data_6[12],data_6[13],data_6[14],data_6[15],data_6[16],data_6[17],data_6[18],data_6[19],data_6[20],data_6[21],data_6[22],data_6[23],data_6[24],data_6[25],data_6[26],data_6[27],data_6[28],data_6[29],data_6[30],data_6[31],data_6[32],data_6[33],data_6[34],data_6[35],data_6[36],data_6[37],data_6[38],data_6[39],data_6[40],data_6[41],data_6[42],data_6[43],data_6[44],data_6[45],data_6[46],data_6[47],data_6[48],data_6[49],data_6[50],data_6[51],data_6[52],data_6[53],data_6[54],data_6[55],data_6[56],data_6[57],data_6[58],data_6[59],data_6[60],data_6[61],data_6[62],data_6[63]))

            print('parameter fc_64_16_6 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_7[0],data_7[1],data_7[2],data_7[3],data_7[4],data_7[5],data_7[6],data_7[7],data_7[8],data_7[9],data_7[10],data_7[11],data_7[12],data_7[13],data_7[14],data_7[15],data_7[16],data_7[17],data_7[18],data_7[19],data_7[20],data_7[21],data_7[22],data_7[23],data_7[24],data_7[25],data_7[26],data_7[27],data_7[28],data_7[29],data_7[30],data_7[31],data_7[32],data_7[33],data_7[34],data_7[35],data_7[36],data_7[37],data_7[38],data_7[39],data_7[40],data_7[41],data_7[42],data_7[43],data_7[44],data_7[45],data_7[46],data_7[47],data_7[48],data_7[49],data_7[50],data_7[51],data_7[52],data_7[53],data_7[54],data_7[55],data_7[56],data_7[57],data_7[58],data_7[59],data_7[60],data_7[61],data_7[62],data_7[63]))

            print('parameter fc_64_16_7 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_8[0],data_8[1],data_8[2],data_8[3],data_8[4],data_8[5],data_8[6],data_8[7],data_8[8],data_8[9],data_8[10],data_8[11],data_8[12],data_8[13],data_8[14],data_8[15],data_8[16],data_8[17],data_8[18],data_8[19],data_8[20],data_8[21],data_8[22],data_8[23],data_8[24],data_8[25],data_8[26],data_8[27],data_8[28],data_8[29],data_8[30],data_8[31],data_8[32],data_8[33],data_8[34],data_8[35],data_8[36],data_8[37],data_8[38],data_8[39],data_8[40],data_8[41],data_8[42],data_8[43],data_8[44],data_8[45],data_8[46],data_8[47],data_8[48],data_8[49],data_8[50],data_8[51],data_8[52],data_8[53],data_8[54],data_8[55],data_8[56],data_8[57],data_8[58],data_8[59],data_8[60],data_8[61],data_8[62],data_8[63]))

            print('parameter fc_64_16_8 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_9[0],data_9[1],data_9[2],data_9[3],data_9[4],data_9[5],data_9[6],data_9[7],data_9[8],data_9[9],data_9[10],data_9[11],data_9[12],data_9[13],data_9[14],data_9[15],data_9[16],data_9[17],data_9[18],data_9[19],data_9[20],data_9[21],data_9[22],data_9[23],data_9[24],data_9[25],data_9[26],data_9[27],data_9[28],data_9[29],data_9[30],data_9[31],data_9[32],data_9[33],data_9[34],data_9[35],data_9[36],data_9[37],data_9[38],data_9[39],data_9[40],data_9[41],data_9[42],data_9[43],data_9[44],data_9[45],data_9[46],data_9[47],data_9[48],data_9[49],data_9[50],data_9[51],data_9[52],data_9[53],data_9[54],data_9[55],data_9[56],data_9[57],data_9[58],data_9[59],data_9[60],data_9[61],data_9[62],data_9[63]))

            print('parameter fc_64_16_9 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_10[0],data_10[1],data_10[2],data_10[3],data_10[4],data_10[5],data_10[6],data_10[7],data_10[8],data_10[9],data_10[10],data_10[11],data_10[12],data_10[13],data_10[14],data_10[15],data_10[16],data_10[17],data_10[18],data_10[19],data_10[20],data_10[21],data_10[22],data_10[23],data_10[24],data_10[25],data_10[26],data_10[27],data_10[28],data_10[29],data_10[30],data_10[31],data_10[32],data_10[33],data_10[34],data_10[35],data_10[36],data_10[37],data_10[38],data_10[39],data_10[40],data_10[41],data_10[42],data_10[43],data_10[44],data_10[45],data_10[46],data_10[47],data_10[48],data_10[49],data_10[50],data_10[51],data_10[52],data_10[53],data_10[54],data_10[55],data_10[56],data_10[57],data_10[58],data_10[59],data_10[60],data_10[61],data_10[62],data_10[63]))

            print('parameter fc_64_16_10 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_11[0],data_11[1],data_11[2],data_11[3],data_11[4],data_11[5],data_11[6],data_11[7],data_11[8],data_11[9],data_11[10],data_11[11],data_11[12],data_11[13],data_11[14],data_11[15],data_11[16],data_11[17],data_11[18],data_11[19],data_11[20],data_11[21],data_11[22],data_11[23],data_11[24],data_11[25],data_11[26],data_11[27],data_11[28],data_11[29],data_11[30],data_11[31],data_11[32],data_11[33],data_11[34],data_11[35],data_11[36],data_11[37],data_11[38],data_11[39],data_11[40],data_11[41],data_11[42],data_11[43],data_11[44],data_11[45],data_11[46],data_11[47],data_11[48],data_11[49],data_11[50],data_11[51],data_11[52],data_11[53],data_11[54],data_11[55],data_11[56],data_11[57],data_11[58],data_11[59],data_11[60],data_11[61],data_11[62],data_11[63]))

            print('parameter fc_64_16_11 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_12[0],data_12[1],data_12[2],data_12[3],data_12[4],data_12[5],data_12[6],data_12[7],data_12[8],data_12[9],data_12[10],data_12[11],data_12[12],data_12[13],data_12[14],data_12[15],data_12[16],data_12[17],data_12[18],data_12[19],data_12[20],data_12[21],data_12[22],data_12[23],data_12[24],data_12[25],data_12[26],data_12[27],data_12[28],data_12[29],data_12[30],data_12[31],data_12[32],data_12[33],data_12[34],data_12[35],data_12[36],data_12[37],data_12[38],data_12[39],data_12[40],data_12[41],data_12[42],data_12[43],data_12[44],data_12[45],data_12[46],data_12[47],data_12[48],data_12[49],data_12[50],data_12[51],data_12[52],data_12[53],data_12[54],data_12[55],data_12[56],data_12[57],data_12[58],data_12[59],data_12[60],data_12[61],data_12[62],data_12[63]))

            print('parameter fc_64_16_12 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_13[0],data_13[1],data_13[2],data_13[3],data_13[4],data_13[5],data_13[6],data_13[7],data_13[8],data_13[9],data_13[10],data_13[11],data_13[12],data_13[13],data_13[14],data_13[15],data_13[16],data_13[17],data_13[18],data_13[19],data_13[20],data_13[21],data_13[22],data_13[23],data_13[24],data_13[25],data_13[26],data_13[27],data_13[28],data_13[29],data_13[30],data_13[31],data_13[32],data_13[33],data_13[34],data_13[35],data_13[36],data_13[37],data_13[38],data_13[39],data_13[40],data_13[41],data_13[42],data_13[43],data_13[44],data_13[45],data_13[46],data_13[47],data_13[48],data_13[49],data_13[50],data_13[51],data_13[52],data_13[53],data_13[54],data_13[55],data_13[56],data_13[57],data_13[58],data_13[59],data_13[60],data_13[61],data_13[62],data_13[63]))

            print('parameter fc_64_16_13 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_14[0],data_14[1],data_14[2],data_14[3],data_14[4],data_14[5],data_14[6],data_14[7],data_14[8],data_14[9],data_14[10],data_14[11],data_14[12],data_14[13],data_14[14],data_14[15],data_14[16],data_14[17],data_14[18],data_14[19],data_14[20],data_14[21],data_14[22],data_14[23],data_14[24],data_14[25],data_14[26],data_14[27],data_14[28],data_14[29],data_14[30],data_14[31],data_14[32],data_14[33],data_14[34],data_14[35],data_14[36],data_14[37],data_14[38],data_14[39],data_14[40],data_14[41],data_14[42],data_14[43],data_14[44],data_14[45],data_14[46],data_14[47],data_14[48],data_14[49],data_14[50],data_14[51],data_14[52],data_14[53],data_14[54],data_14[55],data_14[56],data_14[57],data_14[58],data_14[59],data_14[60],data_14[61],data_14[62],data_14[63]))

            print('parameter fc_64_16_14 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_15[0],data_15[1],data_15[2],data_15[3],data_15[4],data_15[5],data_15[6],data_15[7],data_15[8],data_15[9],data_15[10],data_15[11],data_15[12],data_15[13],data_15[14],data_15[15],data_15[16],data_15[17],data_15[18],data_15[19],data_15[20],data_15[21],data_15[22],data_15[23],data_15[24],data_15[25],data_15[26],data_15[27],data_15[28],data_15[29],data_15[30],data_15[31],data_15[32],data_15[33],data_15[34],data_15[35],data_15[36],data_15[37],data_15[38],data_15[39],data_15[40],data_15[41],data_15[42],data_15[43],data_15[44],data_15[45],data_15[46],data_15[47],data_15[48],data_15[49],data_15[50],data_15[51],data_15[52],data_15[53],data_15[54],data_15[55],data_15[56],data_15[57],data_15[58],data_15[59],data_15[60],data_15[61],data_15[62],data_15[63]))

            print('parameter fc_64_16_15 = 64!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{}{}{}{};'.format(data_16[0],data_16[1],data_16[2],data_16[3],data_16[4],data_16[5],data_16[6],data_16[7],data_16[8],data_16[9],data_16[10],data_16[11],data_16[12],data_16[13],data_16[14],data_16[15],data_16[16],data_16[17],data_16[18],data_16[19],data_16[20],data_16[21],data_16[22],data_16[23],data_16[24],data_16[25],data_16[26],data_16[27],data_16[28],data_16[29],data_16[30],data_16[31],data_16[32],data_16[33],data_16[34],data_16[35],data_16[36],data_16[37],data_16[38],data_16[39],data_16[40],data_16[41],data_16[42],data_16[43],data_16[44],data_16[45],data_16[46],data_16[47],data_16[48],data_16[49],data_16[50],data_16[51],data_16[52],data_16[53],data_16[54],data_16[55],data_16[56],data_16[57],data_16[58],data_16[59],data_16[60],data_16[61],data_16[62],data_16[63]))


        elif i == 3: #bias_fc1 [16]
            print('\n//fc1.bias')
            # print(data)

            data = data.int().tolist()
            # print(data)

            data_1 = data

            print('\nparameter fc_64_16_bias = 16!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{};'.format(data_1[0],data_1[1],data_1[2],data_1[3],data_1[4],data_1[5],data_1[6],data_1[7],data_1[8],data_1[9],data_1[10],data_1[11],data_1[12],data_1[13],data_1[14],data_1[15]))
            # print(data_1[0])

        elif i == 4: #weight_fc2 [10,16]
            print('\n//fc2.weight')
            # print(data)

            data = data.int().tolist()

            data_1 = data[0] #16個 
            data_2 = data[1] #16個
            data_3 = data[2] #16個
            data_4 = data[3] #16個
            data_5 = data[4] #16個
            data_6 = data[5] #16個
            data_7 = data[6] #16個
            data_8 = data[7] #16個
            data_9 = data[8] #16個
            data_10 = data[9] #16個

            print('\nparameter fc_16_10_0 = 16!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{};'.format(data_1[0],data_1[1],data_1[2],data_1[3],data_1[4],data_1[5],data_1[6],data_1[7],data_1[8],data_1[9],data_1[10],data_1[11],data_1[12],data_1[13],data_1[14],data_1[15]))

            print('parameter fc_16_10_1 = 16!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{};'.format(data_2[0],data_2[1],data_2[2],data_2[3],data_2[4],data_2[5],data_2[6],data_2[7],data_2[8],data_2[9],data_2[10],data_2[11],data_2[12],data_2[13],data_2[14],data_2[15]))

            print('parameter fc_16_10_2 = 16!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{};'.format(data_3[0],data_3[1],data_3[2],data_3[3],data_3[4],data_3[5],data_3[6],data_3[7],data_3[8],data_3[9],data_3[10],data_3[11],data_3[12],data_3[13],data_3[14],data_3[15]))

            print('parameter fc_16_10_3 = 16!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{};'.format(data_4[0],data_4[1],data_4[2],data_4[3],data_4[4],data_4[5],data_4[6],data_4[7],data_4[8],data_4[9],data_4[10],data_4[11],data_4[12],data_4[13],data_4[14],data_4[15]))

            print('parameter fc_16_10_4 = 16!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{};'.format(data_5[0],data_5[1],data_5[2],data_5[3],data_5[4],data_5[5],data_5[6],data_5[7],data_5[8],data_5[9],data_5[10],data_5[11],data_5[12],data_5[13],data_5[14],data_5[15]))

            print('parameter fc_16_10_5 = 16!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{};'.format(data_6[0],data_6[1],data_6[2],data_6[3],data_6[4],data_6[5],data_6[6],data_6[7],data_6[8],data_6[9],data_6[10],data_6[11],data_6[12],data_6[13],data_6[14],data_6[15]))

            print('parameter fc_16_10_6 = 16!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{};'.format(data_7[0],data_7[1],data_7[2],data_7[3],data_7[4],data_7[5],data_7[6],data_7[7],data_7[8],data_7[9],data_7[10],data_7[11],data_7[12],data_7[13],data_7[14],data_7[15]))

            print('parameter fc_16_10_7 = 16!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{};'.format(data_8[0],data_8[1],data_8[2],data_8[3],data_8[4],data_8[5],data_8[6],data_8[7],data_8[8],data_8[9],data_8[10],data_8[11],data_8[12],data_8[13],data_8[14],data_8[15]))

            print('parameter fc_16_10_8 = 16!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{};'.format(data_9[0],data_9[1],data_9[2],data_9[3],data_9[4],data_9[5],data_9[6],data_9[7],data_9[8],data_9[9],data_9[10],data_9[11],data_9[12],data_9[13],data_9[14],data_9[15]))

            print('parameter fc_16_10_9 = 16!b{}{}{}{}{}_{}{}{}{}{}_{}{}{}{}{}_{};'.format(data_10[0],data_10[1],data_10[2],data_10[3],data_10[4],data_10[5],data_10[6],data_10[7],data_10[8],data_10[9],data_10[10],data_10[11],data_10[12],data_10[13],data_10[14],data_10[15]))

        else: #bias_fc2 [10]
            print('\n//fc2.bias')
            # print(data)

            data = data.int().tolist()
            # print(data)

            data_1 = data

            print('\nparameter fc_16_10_bias = 10!b{}{}{}{}{}_{}{}{}{}{};'.format(data_1[0],data_1[1],data_1[2],data_1[3],data_1[4],data_1[5],data_1[6],data_1[7],data_1[8],data_1[9]))

