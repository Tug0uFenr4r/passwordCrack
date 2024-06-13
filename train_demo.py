import os
import keyboard
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import base64

plaintexts = open("PasswordDic/2017_top100.txt",'r+').readlines()
encoded_texts = [base64.b64encode(text.replace('\n','').encode()).decode() for text in plaintexts]

# 构建字符映射
all_characters = set(''.join(plaintexts) + ''.join(encoded_texts))
char_to_index = {char: idx for idx, char in enumerate(sorted(all_characters))}
index_to_char = {idx: char for char, idx in char_to_index.items()}

def vectorize_data(texts, char_to_index, max_length):
    data = np.zeros((len(texts), max_length, len(char_to_index)), dtype='float32')
    for i, text in enumerate(texts):
        for t, char in enumerate(text):
            data[i, t, char_to_index[char]] = 1.
    return data

# 数据处理
max_length = max(max(len(text) for text in plaintexts), max(len(text) for text in encoded_texts))
input_data = vectorize_data(plaintexts, char_to_index, max_length)
target_data = vectorize_data(encoded_texts, char_to_index, max_length)


input_data = torch.tensor(input_data)
target_data = torch.tensor(target_data)

# 定义模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        output = self.linear(output)
        return output


def train():
    model = RNNModel(len(char_to_index), 64, len(char_to_index))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if "Athena_final.pth" in os.listdir():
        print("模型存在，开始加载模型")
        model.load_state_dict(torch.load('Athena_final.pth'))
        print("模型加载成功")
        #torch.save(model.state_dict(), 'Athena_init.pth')
    else:
        torch.save(model.state_dict(), 'Athena_final.pth')

    # 训练模型100次
        try:
            print("进入训练")
            for epoch in range(100000):
                optimizer.zero_grad()
                output = model(input_data)
                loss = criterion(output.permute(0, 2, 1), torch.argmax(target_data, dim=2))
                loss.backward()
                optimizer.step()
                print(f'Epoch {epoch+1}: Loss = {loss.item()}')
                if keyboard.is_pressed('q'):
                    print("'q'键被按下，保存模型并退出训练。")
                    torch.save(model.state_dict(), 'model_interrupted.pth')
                    break
            print("训练完成")
            torch.save(model.state_dict(),'Athena_final.pth')
        except KeyboardInterrupt:
            print("训练被手动中断，保存模型。")
            torch.save(model.state_dict(), 'Athena_final.pth')




def decode_sequence(input_seq):
    model = RNNModel(len(char_to_index), 64, len(char_to_index))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.load_state_dict(torch.load('Athena_final.pth'))
    with torch.no_grad():
        input_seq = torch.tensor(input_seq)
        output = model(input_seq)
        predicted_indices = torch.argmax(output, dim=2)
        decoded_sentence = ''.join(index_to_char[int(idx)] for idx in predicted_indices[0])
        return decoded_sentence

def val():
    for i in range(len(plaintexts)):
        input_seq = input_data[i:i + 1]
        print(input_seq)
        decoded_sentence = decode_sequence(input_seq)
        print('Input sentence:', plaintexts[i])
        print('Decoded sentence:', decoded_sentence, '\n')


#train()
val()
#print(plaintexts)
#print(encoded_texts)
#print(all_characters)
#print(char_to_index)
#print(index_to_char)