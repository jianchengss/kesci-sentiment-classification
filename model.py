import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from logger import logger

class SoftMax_CNN(nn.Module):
    def __init__(self, n_feature, n_hidden, n_out, vocb_size):
        super(SoftMax_CNN, self).__init__()
        self.hidden = nn.Linear(n_feature, int(n_feature / 2))
        self.prelu = torch.nn.PReLU()
        self.hidden2 = nn.Linear(int(n_feature / 2), n_hidden)
        self.relu = torch.nn.ReLU()

        self.out = nn.Linear(n_hidden, n_out)

        self.embedding_dim = 10
        self.embeding = nn.Embedding(vocb_size, self.embedding_dim)
        # self.embeding = nn.Embedding(vocb_size, embedding_dim, _weight=embedding_matrix)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                      stride=1, padding=2),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16,64,64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(  # (16,64,64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        cnn_out_dim = 400
        self.out_word = nn.Linear(cnn_out_dim, 2)
        linear_input_dim = n_hidden + cnn_out_dim
        # self.merge_out = nn.Linear(linear_input_dim, int(linear_input_dim / 2))
        # self.relu_out = torch.nn.ReLU()
        # self.final_out = nn.Linear(int(linear_input_dim/ 2), 2)
        logger.info("linear_input_dim: {}".format(linear_input_dim))
        self.final_out = nn.Linear(linear_input_dim, 2)

    def forward(self, x, word_x):
        x = self.hidden(x)
        x = self.relu(x)
        # x = self.prelu(x)
        x = self.hidden2(x)
        x = torch.sigmoid(x)
        # x = self.out(x)

        word_x = self.embeding(word_x)
        # word_x = word_x.unsqueeze(1)
        word_x = word_x.view(1, 1, config.word_max_length, self.embedding_dim)
        # print(x.size())
        word_x = self.conv1(word_x) # cnn_out_dim 400
        # word_x = self.conv2(word_x)
        # word_x = self.conv3(word_x)
        # word_x = self.conv4(word_x)

        word_x = word_x.view(word_x.size(0), -1)  # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）
        # print(x.size())
        # word_x = self.out_word(word_x)

        x = torch.cat([x, word_x], dim=1)
        # x = self.merge_out(x)
        # x = torch.sigmoid(x)
        # x = self.relu_out(x)
        x = self.final_out(x)
        x = F.softmax(x, dim=1)  # 返回的是每个类的概率
        return x



class SoftMax_CNN2(nn.Module):
    def __init__(self, n_feature, n_hidden, n_out, vocb_size):
        super(SoftMax_CNN2, self).__init__()
        self.hidden = nn.Linear(n_feature, int(n_feature / 2))
        self.prelu = torch.nn.PReLU()
        self.hidden2 = nn.Linear(int(n_feature / 2), n_hidden)
        self.relu = torch.nn.ReLU()

        self.linear_out = nn.Linear(n_hidden, n_out)

        self.embedding_dim = 10
        self.embeding = nn.Embedding(vocb_size, self.embedding_dim)
        # self.embeding = nn.Embedding(vocb_size, embedding_dim, _weight=embedding_matrix)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                      stride=1, padding=2),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16,64,64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(  # (16,64,64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out_word = nn.Linear(128, 2)
        linear_input_dim = n_hidden + 128
        # self.merge_out = nn.Linear(linear_input_dim, int(linear_input_dim / 2))
        # self.relu_out = torch.nn.ReLU()
        # self.final_out = nn.Linear(int(linear_input_dim/ 2), 2)
        self.final_out = nn.Linear(linear_input_dim, 2)

    def forward(self, x, word_x):
        x = self.hidden(x)
        x = self.relu(x)
        # x = self.prelu(x)
        x = self.hidden2(x)
        x = torch.sigmoid(x)
        # x = self.out(x)

        x = self.embeding(x)
        # word_x = word_x.unsqueeze(1)
        x = x.view(1, 1, config.word_max_length, self.embedding_dim)
        # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）
        # print(x.size())
        # word_x = self.out_word(word_x)

        # x = torch.cat([x, word_x], dim=1)

        # x = self.merge_out(x)
        # x = torch.sigmoid(x)
        # x = self.relu_out(x)
        x = self.final_out(x)
        x = F.softmax(x, dim=1)  # 返回的是每个类的概率
        return x



class TextCNN(nn.Module):
    def __init__(self, vocb_size, embedding_matrix, embedding_dim):
        super(TextCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeding = nn.Embedding(vocb_size, embedding_dim)
        # self.embeding = nn.Embedding(vocb_size, embedding_dim, _weight=embedding_matrix)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                      stride=1, padding=2),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16,64,64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(  # (16,64,64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(128, 2)

    def forward(self, x):
        x = self.embeding(x)
        x = x.view(x.size(0), 1, config.word_max_length, self.embedding_dim)
        # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）
        # print(x.size())
        x = self.out(x)
        return F.softmax(x, dim=1)  # 返回的是每个类的概率
        # return x




class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args

        embed_num = args.embed_num
        embed_dim = args.embed_dim
        class_num = args.class_num
        Ci = 1
        kernel_num = args.kernel_num
        kernel_sizes = args.kernel_sizes

        self.embed = nn.Embedding(embed_num, embed_dim)

        self.convs_list = nn.ModuleList(
            [nn.Conv2d(Ci, kernel_num, (kernel_size, embed_dim)) for kernel_size in kernel_sizes])

        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, class_num)

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs_list]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        logit = self.fc(x)
        return logit
