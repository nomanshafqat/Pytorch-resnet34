import argparse
import torch
import torch.nn.functional as F
import utils
import dataset
import models
import dataloader
import trainer


parser = argparse.ArgumentParser(description='Pen refinement')

parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.00001)')
parser.add_argument('--dataset-path', default="", help="Directory containing data")

parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='Total number of epochs to run')

parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--cuda', action='store_true', default=True,
                    help=' CUDA training')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='without CUDA training')


args = parser.parse_args()
#args.cuda=False
#args.no_cuda=True
#args.batch_size=50
#args.epochs=1
#args.lr=1e-4
args.momentum=0.9
args.decay=4e-4
args.schedule=[4,8,12]
args.gammas=[0.2, 0.2, 0.2]
#args.dataset_path="/Users/nomanshafqat/Desktop/newdata"

logger = utils.Logger("../", "pens").get_logger()
args.cuda = not args.no_cuda and torch.cuda.is_available()

dataset = dataset.Trendage(args.dataset_path)

train_loader = dataloader.HDDLoader(dataset, dataset.train_data, dataset.train_labels, dataset.transform)
val_loader = dataloader.HDDLoader(dataset, dataset.val_data, dataset.val_labels, dataset.transform)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_iterator = torch.utils.data.DataLoader(train_loader,
                                             batch_size=args.batch_size, shuffle=True, **kwargs)

val_iterator = torch.utils.data.DataLoader(val_loader,
                                             batch_size=args.batch_size, shuffle=True, **kwargs)

# tensor_visualizer = utils.VisualizeTensor()

experiment_model_type = models.model()

if args.cuda:
    experiment_model_type.cuda()
#if not args.no_freeze:
#    utils.freeze_layers(experiment_model_type, args.unfrozen_layers)

print((experiment_model_type.parameters()))

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, experiment_model_type.parameters()), args.lr,
                            momentum=args.momentum, weight_decay=args.decay, nesterov=True)


my_trainer = trainer.Trainer(train_iterator, experiment_model_type, args.cuda, optimizer)

for epoch in range(args.epochs):
    my_trainer.update_lr(epoch, args.schedule, args.gammas)
    my_trainer.train(epoch)
    # my_trainer.evaluate(val_iterator)
    #my_trainer.get_confusion_matrix(val_iterator,2)


    torch.save(my_trainer.model.state_dict(), "../ModelState_"+str(epoch))

