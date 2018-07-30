import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import transforms

from tqdm import tqdm
from torch.autograd import Variable
import logging
import numpy as np
logger = logging.getLogger('trendage')


class Trainer():
    def __init__(self, train_iterator, model, cuda, optimizer=None):
        self.cuda = cuda
        self.train_iterator = train_iterator
        self.model = model
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()

    def update_lr(self, epoch, schedule, gammas):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * gammas[temp]
                    logger.debug("Changing learning rate from %f to %f", self.current_lr,
                                 self.current_lr * gammas[temp])
                    self.current_lr *= gammas[temp]


    def train(self, epoch):

        totensor = transforms.Compose(
            [transforms.ToTensor()])

        self.model.train()
        train_loss = 0
        logger.info("Epoch : %d", epoch)
        for input1, targets, _ in tqdm(self.train_iterator):
            if self.cuda:
                input1, targets = input1.cuda(), targets.cuda()
                self.model.cuda()

            self.optimizer.zero_grad()
            #print("input1=", input1.shape)

            output = self.model(Variable(input1))

            #print("output=", output.shape)

            #print(np.array(targets))

            #print("output=",output[0],targets[0])

            loss = F.mse_loss(output, Variable(targets).float())
            #loss=nn.MSELoss()
            #loss(output,Variable(targets))

            #print("Loss", loss)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(self.train_iterator)
        print(train_loss)

        logger.info("Avg Loss %s", str(train_loss))

    def evaluate(self, iterator):
        """
        Function to evaluate the performance of the trained model.
        :param iterator: Pass the validation set iterator
        :return: N/A; change this to return the accuracy
        TODO:
            Move this to a different evaluator class; we shouldn't need to initialize the trainer just to evaluate our model.
        """

        self.model.eval()
        correct = None
        with torch.no_grad():
            for input1, targets, _ in tqdm(iterator):
                if self.cuda:
                    input1, targets = input1.cuda(), targets.cuda()
                    self.model.cuda()

                output = self.model(Variable(input1))
                print(output)


            logger.info("Val Correct : %s", str(correct))


