from datetime import datetime
import sys, os
sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings(action="ignore")

import argparse
from pprint import pprint
import contextlib
from dataset.transform import *
from dataset.voc import VOCSegmentation as VOC
from torch.utils.data import DataLoader
import os
from model.build_BiSeNet import BiSeNet
import torch
from torch.cuda import amp
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from loss import DiceLoss

# setup the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform():
    train_transform = Compose(
        [
            RandomResizedCrop(320, (0.5, 2.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = Compose(
        [
            PadCenterCrop(size=512),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def val(args, model, dataloader, loss_func):
    # print("start val!")
    # label_info = get_label_info(csv_path)
    tq = tqdm(total=len(dataloader) * args.batch_size)
    tq.set_description("validating:")
    with torch.no_grad():
        model.eval()
        precision_record = []
        loss_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda().long()

            output = model(data)
            loss = loss_func(output, label)
            loss_record.append(loss.item())
            # get RGB predict image
            _, prediction = output.max(dim=1)  # B, H, W
            label = label.cpu().numpy()
            prediction = prediction.cpu().numpy()

            # compute per pixel accuracy
            precision = compute_global_accuracy(prediction, label)
            hist += fast_hist(label.flatten(), prediction.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        tq.close()
        loss_mean = np.mean(loss_record)
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)[:-1]
        miou = np.mean(miou_list)
        return precision, miou, loss_mean


def makeWriter(data, args, model):
    comment = "Optimizer {}, lr {}, batch_size {}".format(
        args.optimizer, args.learning_rate, args.batch_size
    )
    writer = SummaryWriter(comment=comment)
    images, _  = iter(data).next()
    grid = torchvision.utils.make_grid(images)
    images = images.to(device) if args.use_gpu else images
    writer.add_image("images", grid, 0)
    writer.add_graph(model, images)
    return writer

@contextlib.contextmanager
def dummy_cm():
    yield


def train(args, model, optimizer, dataloader_train, dataloader_val, scaler):
    # Prepare the tensorboard
    writer = makeWriter(dataloader_train, args, model)
    # init loss func
    losses = {"dice":DiceLoss(), "crossentropy":torch.nn.CrossEntropyLoss(ignore_index=255)}
    loss_func = losses[args.loss]
    max_miou = 0
    step = 0
    # start training
    for epoch in range(1, args.num_epochs + 1):
        # adjust learning rate
        lr = poly_lr_scheduler(
            optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs
        )
        model.train()
        # lr = args.learning_rate
        loss_record = []
        principal_loss_record = []
        # progress bar
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description("epoch: {}/{}".format(epoch, args.num_epochs))
        for i, (data, label) in enumerate(dataloader_train):
            label = label.type(torch.LongTensor)
            if args.use_gpu:
                data = data.to(device)
                label = label.to(device)

            # forward
            optimizer.zero_grad()
            if scaler:
                cm = amp.autocast()
            else:
                cm = dummy_cm()

            with cm:
                output, output_sup1, output_sup2 = model(data)
                loss1 = loss_func(output, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss = loss1 + loss2 + loss3
            
            tq.update(args.batch_size)
            tq.set_postfix(loss=f"{loss:.4f}", lr=lr)
            # backward
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            step += 1
            # log the progress
            writer.add_scalar("loss_step", loss, step)
            loss_record.append(loss.item())
            principal_loss_record.append(loss1.item())


        tq.close()
        loss_train_mean = np.mean(loss_record)
        pri_train_mean = np.mean(principal_loss_record)
        writer.add_scalar("epoch/loss_epoch_train", float(loss_train_mean), epoch)
        writer.add_scalar("epoch/pri_loss_epoch_train", float(pri_train_mean), epoch)

        if epoch % args.checkpoint_step == 0 or epoch == args.num_epochs:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path, exist_ok = True)
            torch.save(
                model.state_dict(),
                os.path.join(args.save_model_path, f"model_{epoch}_.pth"),
            )

        if (epoch % args.validation_step == 0 or 
            epoch == args.num_epochs or
            epoch == 1):
            precision, miou, val_loss = val(args, model, dataloader_val, loss_func)
            if miou > max_miou:
                max_miou = miou
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_model_path, f"best_{args.loss}_loss.pth"),
                )

            writer.add_scalar("epoch/precision_val", precision, epoch)
            writer.add_scalar("epoch/miou_val", miou, epoch)
            writer.add_scalar("epoch/loss_val", loss, epoch)
            print("epoch: {}, train_loss: {}, val_loss: {}, val_precision: {}, val_miou: {}".format(
                epoch, pri_train_mean, val_loss, precision, miou
            ))
        writer.flush()

    writer.close()


def add_arguments(parser):
    parser.add_argument(
        "--num_epochs", type=int, default=300, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--epoch_start_i",
        type=int,
        default=0,
        help="Start counting epochs from this number",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=10,
        help="How often to save checkpoints (epochs)",
    )
    parser.add_argument(
        "--validation_step",
        type=int,
        default=10,
        help="How often to perform validation (epochs)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of images in each batch"
    )
    parser.add_argument(
        "--context_path",
        type=str,
        default="resnet18",
        help="The context path model you are using, resnet18,resnet50, resnet101.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate used for train"
    )
    parser.add_argument(
        "--data", type=str, default="/gdrive/MyDrive/data", help="path of training data"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="num of workers")
    parser.add_argument(
        "--num_classes", type=int, default=21, help="num of object classes (with void)"
    )
    parser.add_argument(
        "--cuda", type=str, default="0", help="GPU ids used for training"
    )
    parser.add_argument(
        "--use_gpu", type=bool, default=True, help="whether to user gpu for training"
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="path to pretrained model",
    )
    parser.add_argument(
        "--save_model_path", type=str, default="checkpoints", help="path to save model"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer, support rmsprop, sgd, adam",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="crossentropy",
        help="loss function, dice or crossentropy",
    )
    parser.add_argument(
        "--use_amp",
        type=str,
        default="True",
        help="use automatic mixed precision",
    )
    return parser


def get_optim(args, model):
    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        raise ValueError(f"optimizer not supported optimizer: {args.optimizer}")

    return optimizer

def main(params):

    # parse the parameters
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args(params)
    print("Training with following arguments:")
    pprint(vars(args), indent= 4, compact=True)
    print("Running on: {}".format(device if args.use_gpu else torch.device('cpu')))
    # create dataset and dataloader
    train_path = args.data
    train_transform, val_transform = get_transform()
    dataset_train = VOC(train_path, image_set="train", transform=train_transform)
    dataset_val = VOC(train_path, image_set="val", transform=val_transform)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, 
                        shuffle=True, num_workers=args.num_workers, drop_last=True, )
    dataloader_val = DataLoader( dataset_val, batch_size=args.batch_size, 
                        shuffle=True, num_workers=args.num_workers, )

    # build model
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if args.use_gpu:
        model = model.to(device)

    # build optimizer
    optimizer = get_optim(args, model)

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print("load model from %s ..." % args.pretrained_model_path)
        model.load_state_dict(torch.load(args.pretrained_model_path))
        print("Done!")

    scaler = amp.GradScaler() if args.use_amp else None

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val, scaler)
    print("Training completed." , datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))


if __name__ == "__main__":
    params = [
        "--num_epochs", "30",
        "--batch_size", "32",
        "--learning_rate", "0.01",
        "--data", "/root_drive/MyDrive/data" if os.name != 'nt' else 
            r"C:\Users\rehma\Google Drive\data",
        "--num_workers", "8",
        "--validation_step", "2",
        "--num_classes", "21",
        "--cuda", "0",
        "--use_gpu", "True",
        "--save_model_path", "/root_drive/MyDrive/models/res18_30_05_sgd",
        "--context_path", "resnet18",  # set resnet18, resnet50 or resnet101
        "--optimizer", "sgd",
        "--use_amp", "False",
        # "--pretrained_model_path", "/root_drive/MyDrive/models/res18_20_01_sgd/model.pth"
    ]
    print("started:", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    main(params)
    # main(sys.argv[1:])