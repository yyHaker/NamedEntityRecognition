import argparse

import torch
from torch.autograd import Variable
from utils import *

parser = argparse.ArgumentParser(description='LSTM CNN CRF')
parser.add_argument('--epochs', type=int, default=32,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size for training')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', action='store_true',
                    help='enables cuda')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--use-crf', action='store_true',
                    help='use crf')

parser.add_argument('--save', type=str, default='lstm_cnns_crf.pt',
                    help='path to save the final model')
parser.add_argument('--save-epoch', action='store_true',
                    help='save every epoch')
parser.add_argument('--data', type=str, default='corpus.pt',
                    help='location of the data corpus')

parser.add_argument('--char-ebd-dim', type=int, default=32,
                    help='number of char embedding dimension')
parser.add_argument('--kernel-num', type=int, default=32,
                    help='number of kernel')
parser.add_argument('--filter-size', type=int, default=2,
                    help='filter size')
parser.add_argument('--word-ebd-dim', type=int, default=128,
                    help='number of word embedding dimension')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout')
parser.add_argument('--lstm-hsz', type=int, default=256,
                    help='BiLSTM hidden size')
parser.add_argument('--lstm-layers', type=int, default=3,
                    help='biLSTM layer numbers')
parser.add_argument('--l2', type=float, default=0.05,
                    help='l2 regularization')
parser.add_argument('--clip', type=float, default=.5,
                    help='gradient clipping')

args = parser.parse_args()


torch.manual_seed(args.seed)
args.use_cuda = use_cuda = torch.cuda.is_available() and args.cuda_able

# ##############################################################################
# Load data
###############################################################################
from data_loader import DataLoader

data = torch.load(args.data)
args.word_max_len = data["word_max_len"]
args.char_max_len = data["char_max_len"]
args.word_size = data['dict']['word_size']
args.char_size = data['dict']['char_size']
args.label_size = data['dict']['label_size']

training_data = DataLoader(
             data['train']['word'],
             data['train']['char'],
             data['train']['label'],
             args.word_max_len,
             args.char_max_len,
             cuda=use_cuda,
             batch_size=args.batch_size)

validation_data = DataLoader(
              data['valid']['word'],
              data['valid']['char'],
              data['valid']['label'],
              args.word_max_len,
              args.char_max_len,
              batch_size=args.batch_size,
              shuffle=False,
              cuda=use_cuda,
              evaluation=True)

test_data = DataLoader(
    data['test']['word'],
    data['test']['char'],
    data['test']['label'],
    word_max_len=args.word_max_len,
    char_max_len=args.char_max_len,
    cuda=use_cuda,
    batch_size=args.batch_size,
    shuffle=False,
    evaluation=True
)

# ##############################################################################
# Build model
# ##############################################################################
from model import Model
from optim import ScheduledOptim

model = Model(args)
if use_cuda:
   model = model.cuda()

# criterion = torch.nn.CrossEntropyLoss()
optimizer = ScheduledOptim(
            torch.optim.Adam(model.parameters(), lr=args.lr,
                betas=(0.9, 0.98), eps=1e-09, weight_decay=args.l2),
            args.lr)

# ##############################################################################
# Training
# ##############################################################################
import time
from tqdm import tqdm
import const

train_loss = []
valid_loss = []
val_accuracy = []
test_loss = []
test_accuracy = []


def evaluate():
    # set the module to evaluation mode
    model.eval()
    corrects = eval_loss = 0

    for word, char, label in tqdm(validation_data, mininterval=0.2,
                desc='Evaluate Processing', leave=False):
        loss, _ = model(word, char, label)  # (pre_score-label_score).mean()
        pred = model.predict(word, char)

        eval_loss += loss.data[0]
        # every word 被标注正确的个数
        corrects += (pred.data == label.data).sum()
        eval_loss += loss.data

    _size = validation_data.sents_size * args.word_max_len
    return eval_loss[0]/_size, corrects, corrects / _size * 100, _size


def train():
    model.train()
    total_loss = 0
    for word, char, label in tqdm(training_data, mininterval=1,
                desc='Train Processing', leave=True):
        optimizer.zero_grad()
        loss, _ = model(word, char, label)   # loss?
        loss.backward()

        optimizer.step()
        optimizer.update_learning_rate()
        total_loss += loss.data
    return total_loss[0]/training_data.sents_size/args.word_max_len


def test():
    model.eval()
    corrects = test_loss = 0
    for word, char, label in tqdm(test_data, mininterval=0.2, desc='Evaluating Processing',
                                  leave=True):
        loss, _ = model(word, char, label)  # (pre_score-label_score).mean()
        pred = model.predict(word, char)

        corrects += (pred.data == label.data).sum()
        test_loss += loss.data
    _size = test_data.sents_size * args.word_max_len
    return test_loss[0]/_size, corrects, corrects / _size * 100, _size


# ##############################################################################
# Save Model
# ##############################################################################
best_acc = 0.
total_start_time = time.time()

try:
    print('-' * 90)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()  # epoch start time

        loss = train()
        train_loss.append(loss*1000.)

        print('....... done of train epoch {:3d} |cost time: {:2.2f}s | current loss {:5.6f}'.format(epoch, time.time() - epoch_start_time, loss))

        # every epoch done , validation the model
        validation_start_time = time.time()  # validation start time

        loss, corrects, acc, size = evaluate()
        valid_loss.append(loss*1000.)
        val_accuracy.append(acc / 100.)

        print('validation done,  end of epoch {:3d} |cost time: {:2.2f}s | validation loss {:.4f} | '
              'val_accuracy {:.4f}%({}/{})'.format(epoch, time.time()-validation_start_time, loss, acc, corrects, size))

        if not best_acc or corrects > best_acc:
            best_acc = corrects
            # test the model and save the model
            test_start_time = time.time()
            loss, corrects, acc, size = test()
            test_loss.append(loss*1000.)
            test_accuracy.append(acc / 100.)
            print("use current best model to test done, at epoch{:3d} | cost time {:2.2f}s | test loss {:.4f} |"
                  "test_accuracy {:.4f}%({}{})".format(epoch, time.time() - test_start_time, loss, acc, corrects, size))
            # save model parameters
            model_state_dict = model.state_dict()
            model_source = {
                "settings": args,
                "model": model_state_dict,
                "word_dict": data['dict']['word'],
                "label_dict": data['dict']['label']
            }
            torch.save(model_source, args.save)
    print("-"*90)
    # after training save the loss, accuracy for plotting
    print("after training , save the loss, accuracy for plotting.....")
    result_dict = {
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'val_accuracy': val_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }
    save_dict_to_file(result_dict, 'result.pkl')
except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))
