import torch
import pandas as pd
import  os
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import spacy

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary():
    def __init__(self,fre_thre):
        self.itos = {0:'<PAD>',1:'<SOS>',2:'<EOS>',3:"<UNK>"}
        self.stoi = {'<PAD>':0,'<SOS>':1,"<EOS>":2,"<UNK>":3}
        self.fre_thre = fre_thre

    def __len__(self):
        return  len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vacabulary(self,sentence_list):
        frequencis ={}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencis:
                    frequencis[word] =1
                else:
                    frequencis[word] +=1

                if frequencis[word] == self.fre_thre:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx +=1
    def numericalize(self,text):
        tokenize_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenize_text
        ]


class Flickdataset(Dataset):
    def __init__(self,caption_path,image_path,transformer=None,fre_threshold=5):
        self.image_path = image_path
        self.caption_file = pd.read_csv(caption_path)
        self.transform = transformer
        self.img,self.caption = self.caption_file['image'],self.caption_file['caption']

        self.vocab = Vocabulary(fre_threshold)
        self.vocab.build_vacabulary(self.caption.tolist())

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_path,self.img[index]))
        caption = self.caption[index]
        if self.transform is not None:
            img = self.transform(img)

        numericalize_caption =  [self.vocab.stoi['<SOS>']]
        numericalize_caption+= self.vocab.numericalize(caption)
        numericalize_caption.append(self.vacab.stoi["<EOS>"])

        return  img,torch.tensor(numericalize_caption)


class Mycollate:
    def __init__(self,pad_idx):
        self.pad_idx = pad_idx

    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch ]
        imgs - torch.cat(imgs,dim=0)
        targets = [item[1] for item in batch]
        targets= pad_sequence(targets,batch_first=False,padding_value=self.pad_idx)

        return  imgs,targets

def get_loader(
        root_folder,
        caption_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle = True,
        pin_memory = True
):
    dataset = Flickdataset(root_folder,caption_file,transformer=transform)
    pad_idx = dataset.vocab.stoi['<PAD>']

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        pin_memory=pin_memory,
                        collate_fn=Mycollate(pad_idx))

    return  loader

root_dir = 'E:\image captioning\Images'
captions_dir = 'E:\image captioning\captions.txt\captions.txt'
def main():
    dataloader = get_loader(root_dir,
                            caption_file=captions_dir,
                            transform=None)

    for idx,(imgs,captions) in enumerate(dataloader):
        print(imgs.shape)
        print(captions.shape)

if __name__ == "__main__":
    main()