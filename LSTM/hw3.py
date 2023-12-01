import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import data, datasets
import torch.optim as optim
from torchtext.legacy.data import Field
import time
import warnings
warnings.filterwarnings('ignore')


TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)
train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL, train_subtrees=True, filter_pred=lambda ex: ex.label != 'neutral')
TEXT.build_vocab(train_data, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
batch_size = 64
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort_key=lambda x: len(x.text),
    batch_size=batch_size, device=device)

print("vocab finished")


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob):
        super(BiLSTM, self).__init__()
        self.output_size = 1
        self.n_layers = 2
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size,  embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim*2, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        #batch_size = x.size(0)
        x, _ = x
        embeds = self.embedding(x)
        embeds = embeds.permute(1, 0, 2)
        #print(embeds.size())
        lstm_out, hidden = self.lstm(embeds, hidden)

        out = self.dropout(lstm_out)
        out = self.fc(out)

        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size,-1)
        sig_out = sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if (device == "cuda:0"):
            hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_())
        return hidden


vocab_size = len(TEXT.vocab)+1
num_epochs = 10
output_size = 1
embedding_dim = 400 # size of the embeddings
hidden_dim = 256    # Number of units in the hidden layers of our LSTM cells
n_layers = 2        # Number of LSTM layers
dropout_prob = 0.5
skip = 0
path = "./model/biLSTM.pt"


print("init model")
model = BiLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout_prob).to(device)
model.init_hidden(batch_size)
print(model)
criterion = nn.BCELoss()
#criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# 训练模型
print("start training")
for epoch in range(num_epochs):
    print(f"epoch {epoch} start")
    train_loss = 0.0
    train_acc = 0.0
    h = model.init_hidden(batch_size)
    model.train()
    for batch in train_iterator:
        if batch.batch_size != 64:
            skip += batch.batch_size
            continue
        optimizer.zero_grad()
        text, label = batch.text, batch.label
        #print(len(batch))
        #print(len(text[0]))
        h = tuple([each.data for each in h])

        output, h = model(text, h)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        """pred_indices = torch.argmax(output, dim=0)
        correct_predictions = pred_indices.eq(label)
        print(output)
        print(label)
        print(correct_predictions)"""
        train_loss += loss.item() * batch.batch_size
        train_acc += ((output+0.5).view(-1).long() == label).sum()
        #print(train_acc)

    train_loss /= len(train_data)-skip
    train_acc /= len(train_data)-skip

    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

torch.save(model.state_dict(), path)