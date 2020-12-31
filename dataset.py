import os
import json
import pickle
from torch.utils.data import Dataset
import h5py
import numpy as np
import utils
import torch
class Dictionary(object):
    def __init__(self,word2idx=None,idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = {}
        self.word2idx = word2idx
        self.idx2word = idx2word
    #求词典的长度
    @property
    def ntoken(self):
        return len(self.word2idx)
    @property
    def padding_idx(self):
        return len(self.word2idx)

    #向self.idx2worrd中添加词,并返回词word的索引位置
    def add_word(self,word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word)-1
        return self.word2idx[word]

    def tokenize(self,sentence,add_word):

        # :param sentence: 需要向量化的句子
        # :param add_word: 是否需要新添加词
        # :return: 返回词向量表示

        sentence = sentence.lower()
        sentence = sentence.replace(',','').replace('?','').replace('\'s','\'s')
        words = sentence.split()
        tokens = []
        #调用add_word添加词到
        #add_word是bool值
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx.get(w,self.padding_idx-1))
        return tokens

    #将[self.word2idx,self.idx2word]转储到path文件中
    def dump_to_file(self, path):
        pickle.dump([self.word2idx,self.idx2word],open(path,'wb'))
        print('dictionary dumped to %s' %path)

    #将dump_to_file中转存的数据加载出
    @classmethod
    def load_from_file(cls,path):
        print('loading dictionary from %s' %path)
        #word2idx:{'what': 0, 'is': 1, 'this': 2, 'photo': 3, 'taken': 4, 'looking': 5..} 共19901
        #idx2word:['what', 'is', 'this', 'photo', 'taken', 'looking', 'through', 'position'...] 共19901
        word2idx, idx2word = pickle.load(open(path,'rb'))
        d = cls(word2idx,idx2word)
        return d

    def __len__(self):
        return len(self.idx2word)



def _create_entry(img, question, answer):

    # :param img: 图像表示
    # :param question: question.json文件数据
    # :param answer:  annotations.json文件数据
    # :return: 返回用entry字典封装的数据

    if answer is not None:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id':question['question_id'],
        'image_id':question['image_id'],
        'image':img,
        'question':question['question'],
        'answer':answer
    }
    return entry

def _load_dataset(dataroot, name, img_id2val):

    # :param dataroot: 数据集位置
    # :param name: 数据集类型：train,val,test
    # :param img_id2val:所有图像的id
    # :return: 数据条目集：entries

    question_path = os.path.join(dataroot,'Questions','v2_OpenEnded_mscoco_%s2014_questions.json'%name)
    #total:443757 {'image_id': 9, 'question': 'How many cookies can be seen?', 'question_id': 9000}
    questions = sorted(json.load(open(question_path))['questions'],key=lambda x:x['question_id'])
    answer_path = os.path.join(dataroot,'cache','%s_target.pkl'%name)
    #label2ans[17] = '2'
    #answers = total:443757 [{'question_id': 9000, 'image_id': 9, 'labels': [17], 'scores': [1.0]}, {'question_id': 9001, 'image_id': 9, 'labels': [247], 'scores': [0.9]}, ...]
    #raw--question.json:{"image_id": 9, "question": "How many cookies can be seen?", "question_id": 9000} {"image_id": 9, "question": "What color are the dishes?", "question_id": 9001}
    # 9000answer:2, 9001answer:pink and yellow
    answers = pickle.load(open(answer_path,'rb'))
    answers = sorted(answers,key=lambda x:x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = [] #转变成getitem可读的list类型
    for question, answer in zip(questions,answers):
        utils.assert_eq(question['question_id'],answer['question_id']) #比较问题中的question_id是否与答案中一样
        utils.assert_eq(question['image_id'],answer['image_id'])#比较问题中的图像id与答案中是否一样
        img_id = question['image_id']
        #img_id2val[img_id] 获取的是 img_id=9的 图像索引位置:52181
        entries.append(_create_entry(img_id2val[img_id],question,answer))

    return entries


class VQAFeatureDataset(Dataset):

   # VQAFeatureDataset:是模型数据处理的核心类，继承Dataset父类

    def __init__(self, name, dictionary, dataroot='data'):
        super(VQAFeatureDataset,self).__init__()
        assert name in ['train','val'] #判断数据集

        ans2label_path = os.path.join(dataroot,'cache','trainval_ans2label.pkl') #文件位置
        label2ans_path = os.path.join(dataroot,'cache','trainval_label2ans.pkl')
        #self.ans2label = {'net': 0, 'pitcher': 1, 'orange': 2, 'yes': 3, 'white': 4, 'skiing': 5, 'red': 6, ...,} 共3129个单词
        self.ans2label = pickle.load(open(ans2label_path,'rb'))
        #self.label2ans = ['net', 'pitcher', 'orange', 'yes', 'white', 'skiing', 'red', 'frisbee', 'brushing teeth', ...]
        self.label2ans = pickle.load(open(label2ans_path,'rb'))
        self.num_ans_candidates = len(self.ans2label) #答案候选个数 3129
        self.dictionary = dictionary
        #self.img_id2idx = {150367: 0, 283426: 1, 524881: 2, 127298: 3, 232689: 4, 275075: 5, 107610: 6, ...} total:82783
        self.img_id2idx = pickle.load(open(os.path.join(dataroot,'imgids','%s36_imgid2idx.pkl'%name),'rb'))
        print('load features from hdf5 file')
        #加载hdf5文件地址
        h5_path = os.path.join(dataroot,'Bottom-up-features-fixed','%s36.hdf5'%name)
        with h5py.File(h5_path,'r') as hf:
            self.features = np.array(hf.get('image_features')) #图像特征(82783,36,2048)
            self.spatials = np.array(hf.get('spatial_features')) #空间位置 (82783,36,6)

        self.entries = _load_dataset(dataroot,name,self.img_id2idx)

        #向量化坐标
        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(2)
        self.s_dim = self.spatials.size(2)
    #将问题“what is your name” 转换成长度为14的向量表示，不够的用总词数表示（他的代码）
    #不够的此处用0 补齐
    def tokenize(self,max_length=14):
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'],False)
            tokens = tokens[:max_length]
            q_mask = torch.from_numpy((np.arange(max_length) < len(tokens)).astype(int))
            if len(tokens)<max_length:
                padding = [0] *(max_length-len(tokens))
                tokens = tokens+padding
                # q_mask = q_mask+padding

            utils.assert_eq(len(tokens),max_length) #判断向量长度是不是14
            entry['q_token']  = tokens
            entry['q_mask'] = q_mask
    #将numpy类的特征数据转换成tensor数据
    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            q_mask = torch.from_numpy(np.array(entry['q_mask']))
            entry['q_token'] = question
            entry['q_mask'] = q_mask

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'],dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self,index):
        entry = self.entries[index]
        features = self.features[entry['image']] #entry['image']是个特征维度
        spatials = self.spatials[entry['image']] #entry['image']是个特征维度

        f = features.numpy()
        v_mask = (f.sum(0) > 0).astype(int)
        v_mask = torch.from_numpy(v_mask)
        q_mask = entry['q_mask']

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0,labels,scores)

        return features, spatials, question, target

    def __len__(self):
        return len(self.entries)















