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
    #��ʵ�ĳ���
    @property
    def ntoken(self):
        return len(self.word2idx)
    @property
    def padding_idx(self):
        return len(self.word2idx)

    #��self.idx2worrd����Ӵ�,�����ش�word������λ��
    def add_word(self,word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word)-1
        return self.word2idx[word]

    def tokenize(self,sentence,add_word):

        # :param sentence: ��Ҫ�������ľ���
        # :param add_word: �Ƿ���Ҫ����Ӵ�
        # :return: ���ش�������ʾ

        sentence = sentence.lower()
        sentence = sentence.replace(',','').replace('?','').replace('\'s','\'s')
        words = sentence.split()
        tokens = []
        #����add_word��Ӵʵ�
        #add_word��boolֵ
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx.get(w,self.padding_idx-1))
        return tokens

    #��[self.word2idx,self.idx2word]ת����path�ļ���
    def dump_to_file(self, path):
        pickle.dump([self.word2idx,self.idx2word],open(path,'wb'))
        print('dictionary dumped to %s' %path)

    #��dump_to_file��ת������ݼ��س�
    @classmethod
    def load_from_file(cls,path):
        print('loading dictionary from %s' %path)
        #word2idx:{'what': 0, 'is': 1, 'this': 2, 'photo': 3, 'taken': 4, 'looking': 5..} ��19901
        #idx2word:['what', 'is', 'this', 'photo', 'taken', 'looking', 'through', 'position'...] ��19901
        word2idx, idx2word = pickle.load(open(path,'rb'))
        d = cls(word2idx,idx2word)
        return d

    def __len__(self):
        return len(self.idx2word)



def _create_entry(img, question, answer):

    # :param img: ͼ���ʾ
    # :param question: question.json�ļ�����
    # :param answer:  annotations.json�ļ�����
    # :return: ������entry�ֵ��װ������

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

    # :param dataroot: ���ݼ�λ��
    # :param name: ���ݼ����ͣ�train,val,test
    # :param img_id2val:����ͼ���id
    # :return: ������Ŀ����entries

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
    entries = [] #ת���getitem�ɶ���list����
    for question, answer in zip(questions,answers):
        utils.assert_eq(question['question_id'],answer['question_id']) #�Ƚ������е�question_id�Ƿ������һ��
        utils.assert_eq(question['image_id'],answer['image_id'])#�Ƚ������е�ͼ��id������Ƿ�һ��
        img_id = question['image_id']
        #img_id2val[img_id] ��ȡ���� img_id=9�� ͼ������λ��:52181
        entries.append(_create_entry(img_id2val[img_id],question,answer))

    return entries


class VQAFeatureDataset(Dataset):

   # VQAFeatureDataset:��ģ�����ݴ���ĺ����࣬�̳�Dataset����

    def __init__(self, name, dictionary, dataroot='data'):
        super(VQAFeatureDataset,self).__init__()
        assert name in ['train','val'] #�ж����ݼ�

        ans2label_path = os.path.join(dataroot,'cache','trainval_ans2label.pkl') #�ļ�λ��
        label2ans_path = os.path.join(dataroot,'cache','trainval_label2ans.pkl')
        #self.ans2label = {'net': 0, 'pitcher': 1, 'orange': 2, 'yes': 3, 'white': 4, 'skiing': 5, 'red': 6, ...,} ��3129������
        self.ans2label = pickle.load(open(ans2label_path,'rb'))
        #self.label2ans = ['net', 'pitcher', 'orange', 'yes', 'white', 'skiing', 'red', 'frisbee', 'brushing teeth', ...]
        self.label2ans = pickle.load(open(label2ans_path,'rb'))
        self.num_ans_candidates = len(self.ans2label) #�𰸺�ѡ���� 3129
        self.dictionary = dictionary
        #self.img_id2idx = {150367: 0, 283426: 1, 524881: 2, 127298: 3, 232689: 4, 275075: 5, 107610: 6, ...} total:82783
        self.img_id2idx = pickle.load(open(os.path.join(dataroot,'imgids','%s36_imgid2idx.pkl'%name),'rb'))
        print('load features from hdf5 file')
        #����hdf5�ļ���ַ
        h5_path = os.path.join(dataroot,'Bottom-up-features-fixed','%s36.hdf5'%name)
        with h5py.File(h5_path,'r') as hf:
            self.features = np.array(hf.get('image_features')) #ͼ������(82783,36,2048)
            self.spatials = np.array(hf.get('spatial_features')) #�ռ�λ�� (82783,36,6)

        self.entries = _load_dataset(dataroot,name,self.img_id2idx)

        #����������
        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(2)
        self.s_dim = self.spatials.size(2)
    #�����⡰what is your name�� ת���ɳ���Ϊ14��������ʾ�����������ܴ�����ʾ�����Ĵ��룩
    #�����Ĵ˴���0 ����
    def tokenize(self,max_length=14):
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'],False)
            tokens = tokens[:max_length]
            q_mask = torch.from_numpy((np.arange(max_length) < len(tokens)).astype(int))
            if len(tokens)<max_length:
                padding = [0] *(max_length-len(tokens))
                tokens = tokens+padding
                # q_mask = q_mask+padding

            utils.assert_eq(len(tokens),max_length) #�ж����������ǲ���14
            entry['q_token']  = tokens
            entry['q_mask'] = q_mask
    #��numpy�����������ת����tensor����
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
        features = self.features[entry['image']] #entry['image']�Ǹ�����ά��
        spatials = self.spatials[entry['image']] #entry['image']�Ǹ�����ά��

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















