import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        ''' Initialize the layers of this model.'''
        
        # Initially, I didn't defined the super and I was getting error.  
        # AttributeError: cannot assign module before Module.__init__() call
        # I followed the following discussion to solve the error.
        # https://knowledge.udacity.com/questions/71971
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
                
        # word embedding layer to turn vectors into specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # init weights just like we did in Char-Level RNN exercise
        self.init_weights()
        
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        
        # set embed weights
        torch.nn.init.xavier_uniform_(self.fc.weight)
        # Set bias tensor to all zeros
        # self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        torch.nn.init.xavier_uniform_(self.word_embeddings.weight)
        
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        
        # word embedding layer
        captions = self.word_embeddings(captions)
        
        # concatinating the feature vector and captions
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), dim=1)
        # print(inputs.shape)
        
        # LSTM layer
        outputs, _ = self.lstm(inputs)
        
        # FC layer
        outputs = self.fc(outputs)
        
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        predict = []
        
        i = 0
        word_item = None
        
        while i<max_len and word_item !=1:
            outputs, states = self.lstm(inputs, states)
            output = self.fc(outputs)
            
            # word with the max prob
            prob, word = output.max(2)
            # print('Word: ',word)
            # print('Prob: ',prob)
            word_item = word.item()
            predict.append(word_item)
            # Input to next sample is current predicted word
            inputs = self.word_embeddings(word)
            i+=1
        return predict