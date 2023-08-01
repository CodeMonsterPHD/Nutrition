import torch
import json
import csv
import time
import torch.nn as nn
import torch.nn.functional as F

ticks = time.strftime("%Y-%m-%d %H:%M", time.localtime())

class MylossFunc(nn.Module):
    def __init__(self):
        super(MylossFunc, self).__init__()
        self.data = nn.Parameter(data=torch.Tensor(1).float().fill_(0.5))

    def forward(self):
        return self.data

def loss(x,y):
    return torch.sum(x,y)


def train(args,train_loader, model,learnable_pam,optimizer, batch_size, segment,list_anchor):
    model.train()
    loss_section_record = 0
    loss_offset_record = 0
    alpha = args.alpha
    beta = args.beta
    threshold = args.threshold

    for img, img_id, std_mass, section, std_offset in train_loader:
        ####### classify the section of anchor  #######
        new_std_offset = torch.zeros((len(std_offset), 1))
        pre_offset = torch.zeros((len(std_offset), 1))

        optimizer.zero_grad()
        img = img.cuda()
        section = section.cuda()
        std_offset = std_offset.cuda()
        new_std_offset = new_std_offset.cuda()
        pre_offset = pre_offset.cuda()

        pre_section, offset = model(img)
        lambda_n = learnable_pam()
        lambda_n = lambda_n.cuda()

        section = label_smooth(section,segment,threshold)

        # s_max = torch.nn.Softmax()
        # pre_section = s_max(pre_section)

        # matrix_pre_section = struct_matrix(pre_section)
        # matrix_section = struct_matrix(section)

        # loss_section = torch.sum(torch.abs(matrix_pre_section - matrix_section))
        loss_section = -torch.mean(torch.sum(section * F.log_softmax(pre_section, dim=1), dim=1), dim=0)
        ####### classify the section of anchor  #######

        ####### calculate the regression offset #######
        #####  s*e^c   ####
        mass_idx = []

        for i in std_mass:
            for j in range(0,segment):
                if i >= list_anchor[j][0] and i <= list_anchor[j][1]:
                    mass_idx.append(j)
                    break

        for i in range(0,len(mass_idx)):
            new_std_offset[i] = (list_anchor[mass_idx[i]][1] - list_anchor[mass_idx[i]][0])/2 + std_offset[i]
            pre_offset[i] = (list_anchor[mass_idx[i]][1] - list_anchor[mass_idx[i]][0])/2 * torch.exp(lambda_n*offset[i])

        loss_offset = torch.mean(torch.abs(new_std_offset - pre_offset))

        loss = alpha*loss_section + beta*loss_offset
        #loss = loss_offset
        #loss = loss_section

        loss.backward()
        # for name, parms in model.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
        #           ' -->grad_value:', parms.grad)
        optimizer.step()

        loss_section_record = loss_section_record + loss_section.item()
        loss_offset_record = loss_offset_record + loss_offset.item()

    loss_section_avg = loss_section_record / len(train_loader)
    loss_offset_avg = loss_offset_record / len(train_loader)

    return loss_section_avg, loss_offset_avg


def evaluate(test_loader, model,learnable_pam,batch_size,segment,list_anchor,food_name):
    model.eval()
    #path = '/disk/btc010001/ECUSTFD/result/'+ticks + food_name +"_prediction.csv"
    #create_csv(path,food_name)
    list_res = []
    with torch.no_grad():
        for img, img_id, std_mass, section, std_offset in test_loader:
            img = img.cuda()
            pre_section, offset = model(img)
            lambda_n = learnable_pam()
            lambda_n = lambda_n.cuda()

            s_max = torch.nn.Softmax()
            pre_section = s_max(pre_section)

            max_pre_section, idx = torch.max(pre_section, dim=1)

            for i in range(0, len(img_id)):
                left_anchor_set = list_anchor[idx[i]][0]
                section_size = list_anchor[idx[i]][1] - list_anchor[idx[i]][0]

                ############ s*e^c #############
                mass_pre = (left_anchor_set + section_size/2 * torch.exp(lambda_n*offset[i])).item()
                ############ s*e^c #############

                new_prediction = [img_id[i], mass_pre]
                #writer_csv(new_prediction, path)
                list_res.append(new_prediction)
    return list_res

def create_csv(path,food_name):
    with open(path, 'w') as f:
        csv_write = csv.writer(f)
        csv_head = ["id", food_name]
        csv_write.writerow(csv_head)


def writer_csv(prediction, path):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(prediction)


def read_ann_json(file):
    with open(file, 'r') as f:
        datas = json.load(f)
    annotations = datas['annotations']
    return annotations


def all_sum(section,n):
    new_section = torch.clone(section)
    for i in range(0,new_section.shape[1]):
        copy_section = torch.zeros_like(section)
        for j in range(1,n+1):
            # if i == 0:
            #     copy_section[:,i] += new_section[:,i+j]
            # elif i==new_section.shape[1]-1:
            #     copy_section[:,i] += new_section[:,i-j]
            # else:
            if i+j < new_section.shape[1]:
                copy_section[:,i] = copy_section[:,i] + new_section[:,i+j]
            if i-j >=0:
                copy_section[:,i] = copy_section[:,i] + new_section[:,i-j]
        section = section + copy_section
    return section

def normalization(section):
    section = section.type(torch.FloatTensor)
    for i in range(0,section.shape[0]):
        section[i] = section[i]/torch.sum(section[i])  # There will be cuda->cpu
    section = section.cuda()
    return section

def struct_matrix(section):
    copy_section = torch.clone(section)
    new_section = torch.clone(section)
    new_section = new_section.unsqueeze(2)
    new_section = new_section.expand(section.shape[0],section.shape[1],section.shape[1]).clone()
    tmp_section = torch.clone(section[0])
    tmp_section = tmp_section.unsqueeze(0)
    tmp_section = tmp_section.repeat(section.shape[1],1)
    for i in range(0,copy_section.shape[0]):
        for j in range(0,copy_section.shape[1]):
           tmp_section[j] = copy_section[i] + section[i][j]
        new_section[i] = tmp_section

    section = new_section

    return section

def label_smooth(section,segment,tau):
    # section = section.type(torch.FloatTensor)
    # c = segment
    # new_value1 = 1-tau
    # new_value0 = float(tau)/float(c-1)
    # for i in range(0,section.shape[0]):
    #     for j in range(0,section.shape[1]):
    #         if section[i][j] == 1:
    #             section[i][j] = new_value1
    #         else:
    #             section[i][j] = new_value0
    section = section.type(torch.FloatTensor)
    c = segment
    new_value1 = 1 - tau
    for i in range(0, section.shape[0]):
        for j in range(0, section.shape[1]):
            if section[i][j] == 1:
                if j - 1 >= c - j - 1:
                    m = j - 1
                    section[i][j] = new_value1
                    for k in range(0, j):
                        section[i][k] = k + 1
                    for k in range(j + 1, section.shape[1]):
                        section[i][k] = m - (k - 1 - m)
                else:
                    m = c - j - 1
                    section[i][j] = new_value1
                    tmp_right = m
                    tmp_left = 1
                    for k in range(0, j):
                        section[i][k] = tmp_left
                        tmp_left = tmp_left + 1
                    for k in range(j + 1, section.shape[1]):
                        section[i][k] = tmp_right
                        tmp_right = tmp_right - 1
                sum_n = 0
                for k in range(0, section.shape[1]):
                    if k != j:
                        sum_n = sum_n + section[i][k]
                for k in range(0, section.shape[1]):
                    if k != j:
                        section[i][k] = section[i][k] / sum_n * tau
                break
    section = section.cuda()
    return section


if __name__ == '__main__':
    # a = torch.tensor([[1,2,3],[2,3,4]])
    # b = torch.tensor([[[1, 2,3], [2,3,4]], [[-1, -2,-3], [-2, -3,-4]],[[-1, -2,-3], [-2, -3,-4]]])
    # a = b
    # print(a.shape,b.shape)
    # c = torch.mul(a,b)
    # print(c.shape)
    # print(c)
    # c = torch.einsum('ijk,ijl->ijl', a, b)
    # print(c.shape)
    # print(c)

    # section = torch.tensor([0.1,0.05,0.05,0.05,0.4,0.05,0.1,0.1,0.05,0.05])
    # true_section = torch.tensor([0,0,0,0,0,0,1,0,0,0])
    # section = section.resize(1,10)
    # true_section = true_section.resize(1, 10)
    # section = all_sum(section,1)
    # true_section = all_sum(true_section, 1)
    # print(section,true_section)
    # section = all_sum(section,2)
    # true_section = all_sum(true_section, 2)
    # print(section,true_section)
    # section = all_sum(section,3)
    # true_section = all_sum(true_section, 3)
    # print(section,true_section)
    # section = all_sum(section,4)
    # true_section = all_sum(true_section, 4)
    # print(section,true_section)
    # section = normalization(section)

    section = torch.tensor([[0.1,0.05,0.05,0.1,0.4,0.1,0.1,0.1],
                            [0.5,0.05,0.1,0.05,0.05,0.05,0.1,0.1]])
    matrix_section = struct_matrix(section)
    print(matrix_section)



