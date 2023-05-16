import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch_optimizer as optim2
import argparse
import time
from quantize import*
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#ArgumentParserはプログラム実行時にコマンドラインで引数を受け取る処理を簡単に実装できる標準ライブラリ
#コマンドライン引数とはmain 関数の引数のこと
parser = argparse.ArgumentParser(description = 'setting') #パーサーを作る
#parser.add_argumentで受け取る引数を追加していく
parser.add_argument('--data_dir', type = str, default = './mnist') 
parser.add_argument('--batch_size', type =int, default = 100) #バッチサイズ
parser.add_argument('--lr', type = int, default = 0.001) #学習率
parser.add_argument('--epochs', type = int, default = 1) #エポック数
parser.add_argument('--w_bit', type = int, default = 1) #重み
parser.add_argument('--a_bit', type = int, default = 3) #出力
args = parser.parse_args(args = []) #引数の解析

#データの保存先
# PATH_acc = "./acc_adam_round(2^k-1)_1_3_decimal0.text"
# PATH_acc = "./acc_adam_round(2^k-1)_1_3_decimal1.text"
# PATH_acc = "./acc_adam_round(2^k-1)_1_3_decimal2.text"
# PATH_acc = "./acc_adam_round(2^k-1)_1_3_decimal3.text"
# PATH_acc = "./acc_adam_round(2^k)_1_3_decimal0.text"
# PATH_acc = "./acc_adam_round(2^k)_1_3_decimal1.text"
# PATH_acc = "./acc_adam_round(2^k)_1_3_decimal2.text"
# PATH_acc = "./acc_adam_round(2^k)_1_3_decimal3.text"

PATH_acc = "./acc_adam_mix(2^k-1)_1_3.text"
# PATH_acc = "./acc_adam_mix(2^k)_1_3.text"

# PATH_acc = "./acc_adabound_round(2^k-1)_1_3_decimal0.text"
# PATH_acc = "./acc_adabound_round(2^k-1)_1_3_decimal1.text"
# PATH_acc = "./acc_adabound_round(2^k-1)_1_3_decimal2.text"
# PATH_acc = "./acc_adabound_round(2^k-1)_1_3_decimal3.text"
# PATH_acc = "./acc_adabound_round(2^k)_1_3_decimal0.text"
# PATH_acc = "./acc_adabound_round(2^k)_1_3_decimal1.text"
# PATH_acc = "./acc_adabound_round(2^k)_1_3_decimal2.text"
# PATH_acc = "./acc_adabound_round(2^k)_1_3_decimal3.text"

# PATH_acc = "./acc_adam_floor(2^k-1)_1_3_decimal0.text"
# PATH_acc = "./acc_adam_floor(2^k-1)_1_3_decimal1.text"
# PATH_acc = "./acc_adam_floor(2^k-1)_1_3_decimal2.text"
# PATH_acc = "./acc_adam_floor(2^k-1)_1_3_decimal3.text"
# PATH_acc = "./acc_adam_floor(2^k)_1_3_decimal0.text"
# PATH_acc = "./acc_adam_floor(2^k)_1_3_decimal1.text"
# PATH_acc = "./acc_adam_floor(2^k)_1_3_decimal2.text"
# PATH_acc = "./acc_adam_floor(2^k)_1_3_decimal3.text"

# PATH_acc = "./acc_adabound_floor(2^k-1)_1_3_decimal0.text"
# PATH_acc = "./acc_adabound_floor(2^k-1)_1_3_decimal1.text"
# PATH_acc = "./acc_adabound_floor(2^k-1)_1_3_decimal2.text"
# PATH_acc = "./acc_adabound_floor(2^k-1)_1_3_decimal3.text"
# PATH_acc = "./acc_adabound_floor(2^k)_1_3_decimal0.text"
# PATH_acc = "./acc_adabound_floor(2^k)_1_3_decimal1.text"
# PATH_acc = "./acc_adabound_floor(2^k)_1_3_decimal2.text"
# PATH_acc = "./acc_adabound_floor(2^k)_1_3_decimal3.text"

# PATH_acc = "./acc_round(2^k-1)_3_3.text"
# PATH_acc = "./acc_round(2^k)_3_3.text"
# PATH_acc = "./acc_floor(2^k-1)_3_3.text"
# PATH_acc = "./acc_floor(2^k)_3_3.text"

# PATH_model = "./mnist_adam_round(2^k-1)_1_3_decimal0.pt"
# PATH_model = "./mnist_adam_round(2^k-1)_1_3_decimal1.pt"
# PATH_model = "./mnist_adam_round(2^k-1)_1_3_decimal2.pt"
# PATH_model = "./mnist_adam_round(2^k-1)_1_3_decimal3.pt"
# PATH_model = "./mnist_adam_round(2^k)_1_3_decimal0.pt"
# PATH_model = "./mnist_adam_round(2^k)_1_3_decimal1.pt"
# PATH_model = "./mnist_adam_round(2^k)_1_3_decimal2.pt"
# PATH_model = "./mnist_adam_round(2^k)_1_3_decimal3.pt"

PATH_model = "./mnist_adam_mix(2^k-1)_1_3.pt"
# PATH_model = "./mnist_adam_mix(2^k)_1_3.pt"

# PATH_model = "./mnist_adabound_round(2^k-1)_1_3_decimal0.pt"
# PATH_model = "./mnist_adabound_round(2^k-1)_1_3_decimal1.pt"
# PATH_model = "./mnist_adabound_round(2^k-1)_1_3_decimal2.pt"
# PATH_model = "./mnist_adabound_round(2^k-1)_1_3_decimal3.pt"
# PATH_model = "./mnist_adabound_round(2^k)_1_3_decimal0.pt"
# PATH_model = "./mnist_adabound_round(2^k)_1_3_decimal1.pt"
# PATH_model = "./mnist_adabound_round(2^k)_1_3_decimal2.pt"
# PATH_model = "./mnist_adabound_round(2^k)_1_3_decimal3.pt"

# PATH_model = "./mnist_adam_floor(2^k-1)_1_3_decimal0.pt"
# PATH_model = "./mnist_adam_floor(2^k-1)_1_3_decimal1.pt"
# PATH_model = "./mnist_adam_floor(2^k-1)_1_3_decimal2.pt"
# PATH_model = "./mnist_adam_floor(2^k-1)_1_3_decimal3.pt"
# PATH_model = "./mnist_adam_floor(2^k)_1_3_decimal0.pt"
# PATH_model = "./mnist_adam_floor(2^k)_1_3_decimal1.pt"
# PATH_model = "./mnist_adam_floor(2^k)_1_3_decimal2.pt"
# PATH_model = "./mnist_adam_floor(2^k)_1_3_decimal3.pt"

# PATH_model = "./mnist_adabound_floor(2^k-1)_1_3_decimal0.pt"
# PATH_model = "./mnist_adabound_floor(2^k-1)_1_3_decimal1.pt"
# PATH_model = "./mnist_adabound_floor(2^k-1)_1_3_decimal2.pt"
# PATH_model = "./mnist_adabound_floor(2^k-1)_1_3_decimal3.pt"
# PATH_model = "./mnist_adabound_floor(2^k)_1_3_decimal0.pt"
# PATH_model = "./mnist_adabound_floor(2^k)_1_3_decimal1.pt"
# PATH_model = "./mnist_adabound_floor(2^k)_1_3_decimal2.pt"
# PATH_model = "./mnist_adabound_floor(2^k)_1_3_decimal3.pt"

# PATH_model = "./mnist_round(2^k-1)_3_3.pt"
# PATH_model = "./mnist_round(2^k)_3_3.pt"
# PATH_model = "./mnist_floor(2^k-1)_3_3.pt"
# PATH_model = "./mnist_floor(2^k)_3_3.pt"

print('GPU')
print(torch.cuda.is_available()) #PyTorchでGPUが使用可能か確認,使用できればTrue、できなければFalseを返す。
print(torch.cuda.get_device_name(0)) #GPUの名称をprint
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #deviceに、計算に用いるデバイス(ここではGPU)をいれる

#訓練データ
train_data = MNIST(root = args.data_dir, train = True, download = True, transform = transforms.ToTensor())
train_data_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = 0 ,pin_memory = True)
#テストデータ
test_data = MNIST(root = args.data_dir, train =False, download = True, transform = transforms.ToTensor())
test_data_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, num_workers = 0, pin_memory = True)

#CNNのモデル
class Net(nn.Module): #Netclassがnn.Moduleというclassを継承している
    def __init__(self): #コンストラクタ
        super().__init__() #super(Net,self)じゃなくてもいい??
        BinaryConv2d = conv2d_Q_fn(args.w_bit) #量子化畳み込み ここで重みの量子化も行われている
        BinaryLinear = linear_Q_fn(args.w_bit) #量子化全結合層 ここで重みの量子化も行われている
        self.act = activation_quantize_fn(args.a_bit) #出力の量子化
        self.conv1 = BinaryConv2d(1, 1, 5, bias = False) #(その入力のチャネル数,畳み込み後のチャネル数,畳み込みをするための正方形フィルタ(カーネル)の1辺のサイズ)
        self.conv2 = BinaryConv2d(1, 4, 5, bias = False) 
        self.fc1 = BinaryLinear(4*4*4, 16, bias = True) #(各入力サンプルのサイズ,各出力サンプルのサイズ)
        self.fc2 = BinaryLinear(16, 10, bias = True) #MNISTは0~9までの10分類なので出力10
        self.soft = nn.Softmax(dim = 1) #活性化関数(softmax) dim=1のとき行単位でsoftmaxをかける 各行合計が1

    def forward(self, x): #入力x
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

net = Net().to(device) #インスタンス化
print('\n**********ネットワーク概要**********')
print(net)
criterion = nn.CrossEntropyLoss() #誤差関数
#最適化手法(Adam)
optimizer = optim.Adam(net.parameters(), lr = args.lr, betas = [0.9, 0.999], eps = 0.00000001) 
# optimizer = optim2.AdaBound(net.parameters(), lr = args.lr, betas = [0.9, 0.999], eps = 0.00000001,weight_decay=0,amsbound=False) 
# optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum=0) 

acc_part = []
acc_data = []
loss_data = []
loss_all_data = []
for i in range(10):
    print(f"-------{i+1}回目---------")
    epochs = args.epochs
    #訓練 誤差逆伝播法
    for epoch in range(epochs):
        running_loss = 0.0
        for j,(data, labels) in enumerate(train_data_loader): #train_data_loaderは全dataと対応するlabelをBatch毎に分けてセットで持ち,その1セットを変数detaとlabelに渡している. enumerateは各要素にインデックスをつける
            train_data, teacher_labels = data.to(device), labels.to(device) #取得したdataとlabelをGPUで使用するようにする.
            optimizer.zero_grad() #最適化対象のすべてのテンソルの勾配を0で初期化 この勾配情報とは後のbackward()という関数により求められるもので,パラメータを更新する際に使用する.iterationのたびに前の勾配情報が残らないようにするため0にする.
            outputs = net(train_data) #インスタンスnetに引数train_dataをいれる
            loss = criterion(outputs,teacher_labels) #outputsとlabelsとのlossを計算 学習をするための罰則の値を決める
            loss.backward() #計算したlossから勾配情報を計算 この勾配情報を用いてcriterionのlossが小さくなるように更新をしていく
            optimizer.step() #学習率と最適化手法に基づいて実際に勾配情報を使用してパラメータを更新
            running_loss += loss.item() #lossを足していく
            loss_part = running_loss/100
            # print(j)
            if j % 100 == 99: #100ごとに
                print("epoch : {0} [{1}/60000]   loss : {2}".format(epoch + 1, (j + 1)*100, loss_part))
                loss_data.append(loss_part)
                if i == 9:
                    # with open(f'./mnist_loss_adam_round(2^k-1)_decimal0.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_round(2^k-1)_decimal1.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_round(2^k-1)_decimal2.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_round(2^k-1)_decimal3.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_round(2^k)_decimal0.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_round(2^k)_decimal1.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_round(2^k)_decimal2.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_round(2^k)_decimal3.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)

                    # with open(f'./mnist_loss_adabound_round(2^k-1)_decimal0.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_round(2^k-1)_decimal1.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_round(2^k-1)_decimal2.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_round(2^k-1)_decimal3.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_round(2^k)_decimal0.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_round(2^k)_decimal1.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_round(2^k)_decimal2.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_round(2^k)_decimal3.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)

                    # with open(f'./mnist_loss_adam_floor(2^k-1)_decimal0.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_floor(2^k-1)_decimal1.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_floor(2^k-1)_decimal2.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_floor(2^k-1)_decimal3.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_floor(2^k)_decimal0.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_floor(2^k)_decimal1.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_floor(2^k)_decimal2.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_floor(2^k)_decimal3.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)

                    # with open(f'./mnist_loss_adabound_floor(2^k-1)_decimal0.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_floor(2^k-1)_decimal1.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_floor(2^k-1)_decimal2.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_floor(2^k-1)_decimal3.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_floor(2^k)_decimal0.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_floor(2^k)_decimal1.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_floor(2^k)_decimal2.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adabound_floor(2^k)_decimal3.pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)

                    with open(f'./mnist_loss_adam_mix(2^k-1).pickle', 'wb') as f:
                            pickle.dump(loss_data, f)
                    # with open(f'./mnist_loss_adam_mix(2^k).pickle', 'wb') as f:
                    #         pickle.dump(loss_data, f)
                    
                running_loss = 0.0
    print("finish\n")
    count = 0
    total = 0

    #テスト
    t1 = time.time() #時間測る
    for k,(data, labels) in enumerate(test_data_loader):
        test_data, teacher_labels = data.to(device), labels.to(device)
        results = net(test_data) #インスタンスnetに引数test_dataをいれる
        _, predicted = torch.max(results.data, 1) #result.dataの行ごとの最大値の添え字(予想位置)取得 
        count += (predicted == teacher_labels).sum() #予想位置と実際の正解を比べ,正解している数だけ足す
        total += teacher_labels.size(0) #labelの数を足していくことでデータの総和を取る
        acc_part.append((int(count) / int(total)) * 100) #各正答率計算
    # with open(f'./mnist_acc_adam_round(2^k-1)_decimal0.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_round(2^k-1)_decimal1.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_round(2^k-1)_decimal2.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_round(2^k-1)_decimal3.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_round(2^k)_decimal0.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_round(2^k)_decimal1.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_round(2^k)_decimal2.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_round(2^k)_decimal3.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)

    # with open(f'./mnist_acc_adabound_round(2^k-1)_decimal0.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_round(2^k-1)_decimal1.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_round(2^k-1)_decimal2.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_round(2^k-1)_decimal3.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_round(2^k)_decimal0.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_round(2^k)_decimal1.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_round(2^k)_decimal2.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_round(2^k)_decimal3.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    

    # with open(f'./mnist_acc_adam_floor(2^k-1)_decimal0.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_floor(2^k-1)_decimal1.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_floor(2^k-1)_decimal2.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_floor(2^k-1)_decimal3.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_floor(2^k)_decimal0.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_floor(2^k)_decimal1.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_floor(2^k)_decimal2.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_floor(2^k)_decimal3.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)

    # with open(f'./mnist_acc_adabound_floor(2^k-1)_decimal0.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_floor(2^k-1)_decimal1.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_floor(2^k-1)_decimal2.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_floor(2^k-1)_decimal3.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_floor(2^k)_decimal0.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_floor(2^k)_decimal1.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_floor(2^k)_decimal2.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adabound_floor(2^k)_decimal3.pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)

    with open(f'./mnist_acc_adam_mix(2^k-1).pickle', 'wb') as f:
            pickle.dump(acc_part, f)
    # with open(f'./mnist_acc_adam_mix(2^k).pickle', 'wb') as f:
    #         pickle.dump(acc_part, f)
    
    t2 = time.time()
    elapsed_time = t2 - t1 #かかった時間

    accuracy = (int(count) / int(total)) * 100 #正答率計算

    #結果表示
    print(f'実行時間 : {elapsed_time}[s]')
    print(f'accuracy : {accuracy}[%]\n')
    acc_data.append(accuracy)
    EPOCH = args.epochs
    W_BIT = args.w_bit
    A_BIT = args.a_bit
    file = open(PATH_acc, 'w')
    file.write(f'epoch : {EPOCH}\n')
    file.write(f'重み : {W_BIT}\n')
    file.write(f'出力 : {A_BIT}\n')
    file.write(f'実行時間 : {elapsed_time}[s]\n')
    file.write(f'accuracy : {accuracy}[%]')
    file.close

    if accuracy > preacc:
        torch.save(net.state_dict(), PATH_model) # モデルを保存する。

        net.load_state_dict(torch.load(PATH_model)) # 保存したモデルを読み込む。

        # モデルの state_dict のsizeを表示
        print("Model's state_dict:")
        for key, param in net.state_dict().items():
            print(key, "\t", param.size())

        model_data = net.state_dict()
        print('\n**********重み**********')
        print(model_data) #モデルの中身を表示

        # オプティマイザの state_dict を表示
        print("Optimizer's state_dict:")
        for key, param in optimizer.state_dict().items():
            print(key, "\t", param)
        
        preacc = accuracy

    
print(acc_data)
print(sum(acc_data)/len(acc_data))