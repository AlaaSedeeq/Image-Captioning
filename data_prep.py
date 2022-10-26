import os
import nltk
import torch
import numpy as np
import pandas as pd
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence 
from torchvision.transforms import transforms


###############################
# Caption & Image preparation #
###############################
class CaptionPrep:
    '''
    Prepare images caption before feeding into the RNN
    - START (start of sentence)
    - END (end of sentence)
    - PAD (for batch padding)
    - UNK (unknown) for token doesn't exist in the vocabulary 
    ...
    Attributes
    ----------
    freq_th : Number of occurance for a token in order to consider in the vocabulary 
    ''' 
    def __init__(self, freq_th:int):
        # special tokens
        # PAD, UNK, START, END = '#PAD#', '#UNK#', '#START#', '#END#'
        self.idxtos = {0:'#START#', 1:'#END#', 2:'#UNK#', 3:'#PAD#'}
        self.stoidx = None
        self.freq_th = freq_th
    
    def Build_Dictionary(self, corpus:list):    
        """
        Create a  dictionary {token: index} for all train data unique tokens that has occurance value >= thresholds
        """
        tokens = nltk.tokenize.word_tokenize(' '.join(list(map(lambda x: x.lower(),corpus))))
        freq = nltk.FreqDist(tokens)
        self.idxtos.update({i+len(self.idxtos): j.lower()\
                            for i,(j,k) in enumerate(dict(freq).items()) if k>self.freq_th})
        self.stoidx = dict(zip(self.idxtos.values(), self.idxtos.keys()))
        self.vocab_size = len(self.stoidx)
    
    def Sent_idx(self, sent):
        """
        Returns the sentence (image caption) after converting it into numeric values and adding the EOS & SOS
        """
        tok = nltk.tokenize.word_tokenize(sent)
        # adding start-of-sentence and end-of-sentence
        return [self.stoidx['#START#']] +\
               [self.stoidx[w.lower()] if w.lower() in self.stoidx else self.stoidx['#UNK#'] for w in tok] +\
               [self.stoidx['#END#']]


###########################
# Pytorch Dataste creator #
###########################
class ImgCapData(Dataset):
    '''
    Create torch Dataset for the data loader
    ...
    Attributes
    ----------
    img_path: image data path
    cap_path: images captions path
    freq_th: Number of occurance for a token in order to consider in the vocabulary
    transform: Data transformation to apply on the data
    ''' 
    def __init__(self, img_path, cap_path, freq_th, transform=None, target_transform=None):
        self.img_path = img_path
        self.cap_path = cap_path
        self.transform = transform
        self.target_transform = target_transform
        self.freq_th = freq_th
        # it's a comma separated with two columns image & caption
        # split images names and captions
        data = pd.read_csv(self.cap_path)
        data['caption'] = data['caption'].apply(lambda x: x.strip())
        self.captions = data['caption']
        self.img_name = data['image']
        self.len = len(data)
        
        # prepare captions before feeding into the Network
        self.cap_prep = CaptionPrep(self.freq_th)
        self.cap_prep.Build_Dictionary(self.captions.tolist())
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        
        image_name = self.img_name[idx]
        image = np.array(Image.open(os.path.join(self.img_path, image_name)).convert("RGB"))
                
        if self.transform is not None:
            image = torch.tensor(self.transform(image))
            
#         if self.target_transform is not None:
#             label = self.target_transform(label)

        caption = torch.tensor(self.cap_prep.Sent_idx(self.captions[idx]))
        
        return image, caption
    
################################
# Data Loader collate fnnction #
################################
class Collate_fun:
    '''
    Customized dataloader data collate function for to ensure sequence(caption) padding
    ...
    Attributes
    ----------
    pad_idx: index to put in the sequence as a padding value
    max_len: maximum sequence length
    '''
    def __init__(self, pad_idx, max_len=30):
        self.pad_idx = pad_idx
        self.max_len = max_len

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        captions = [item[1] for item in batch]
#         if not self.max_len:
#             self.max_len = len(max(captions, key=lambda x: x.shape[0]))
#         else:
#             self.max_len = min(self.max_len, len(max(captions, key=lambda x:x.shape[0])))
#         captions = torch.Tensor([[idx for i, idx in enumerate(cap[:self.max_len])] + [self.pad_idx]*(max(self.max_len-len(cap),0))\
#                          for cap in captions]).int()
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return imgs, captions

class data_prep:
    '''
    Prepare images & captions dataset and create pytorch dataloader
    ...
    Attributes
    ----------
    img_path: images data path 
    cap_path: captions data path
    freq_th: no. of occurrence for a word in the corpus to be considered in the dictionary
    transformer: images data transformer
    inv_trans: image inverse transformer
    target_transform: caption transformer
    pin_memory: pin dataloader to a device 
    batch_size: batch size 
    shuffle: data shuffling
    '''
    def __init__(
        self, 
        img_path, 
        cap_path, 
        freq_th=0, 
        transformer=None, 
        inv_trans=None,
        target_transform=None, 
        pin_memory=True,
        batch_size=32, 
        shuffle=True
    ):
        self.img_path = img_path
        self.cap_path = cap_path
        self.freq_th = freq_th
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_transform = target_transform
        self.pin_memory = pin_memory
        
        # Transformer        
        if transformer is None and inv_trans is None:
            self.transformer = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((356, 356)),
                transforms.RandomCrop((350, 350)),
                transforms.CenterCrop(299),
                transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], 
                    std=[0.5, 0.5, 0.5])
            ])

            self.inv_trans = transforms.Compose([
                transforms.Normalize(
                    mean = [ 0., 0., 0. ],
                    std = [ 1/0.5, 1/0.5, 1/0.5 ]
                ),
                transforms.Normalize(
                    mean = [ -0.5, -0.5, -0.5 ],
                    std = [ 1., 1., 1. ])
            ])

    def get_data(self):
        '''
        Returns pytorch dataset and dataloader
        '''
        # Gathering Data
        data = ImgCapData(
            img_path=self.img_path,
            cap_path=self.cap_path,
            freq_th=self.freq_th,
            transform=self.transformer,
            target_transform=self.target_transform
        )
        
        self.collate_fn = Collate_fun(pad_idx=data.cap_prep.stoidx['#PAD#'])
        self.vocab_size = data.cap_prep.vocab_size
        self.idxtos = data.cap_prep.idxtos
        self.stoidx = data.cap_prep.stoidx

        # Create DataLoader
        dataloader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
        
        return data, dataloader
