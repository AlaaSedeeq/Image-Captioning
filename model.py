import torch 
import torch.nn as nn
import torch.nn .functional as F
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, feature_size, enc_dropout, pre_trained=None):
        super(Encoder, self).__init__()
        self.feature_size = feature_size
        self.dropout = enc_dropout
        self._feature_extr = pre_trained if pre_trained is not None\
                             else models.inception_v3(pretrained=True, aux_logits=False)
        for param in self._feature_extr.parameters():
            param.requires_grad = True
            param.aux_logits=False
        self._feature_extr.fc = nn.Linear(self._feature_extr.fc.in_features, self.feature_size)
        self._act = nn.ReLU()
        self._dropout = nn.Dropout(self.dropout)
        
    def forward(self, x):
        return torch.rand(len(x), self.feature_size)
        out = self._feature_extr(x)[0]
        out = self._act(out)
        out = self._dropout(out)
        return out
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, num_layers, embed_size, hidden_size, dec_dropout):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.dropout = dec_dropout
        
        self.embed_layer = nn.Embedding(vocab_size, self.embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.Dropout = nn.Dropout(self.dropout)
        
    def forward(self, features, caption):
        cap_embed = self.embed_layer(caption)
        cap_embed = self.Dropout(cap_embed).view(cap_embed.shape[0], -1)
        cap_embed = cap_embed.view(cap_embed.shape[0], -1)
        lstm_in = torch.cat((features, cap_embed), dim=1)
        lstm_hidden, lstm_out = self.lstm(lstm_in)
        out = self.linear(lstm_hidden)
        return out



class Encoder_Decoder(nn.Module):
    def __init__(self, embed_size, enc_dropout, hidden_size, vocab_size, num_layers, dec_dropout):
        super(CNNtoRNN, self).__init__()
        self.Encoder = Encoder(embed_size, enc_dropout, pre_trained=None)
        self.Decoder = Decoder(vocab_size, num_layers, embed_size, hidden_size, dec_dropout)

    def forward(self, images, captions):
        features = self.Encoder(images)
        outputs = self.Decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == '#EOS#':
                    break

        return [vocabulary.itos[idx] for idx in result_caption]