from loss import DiceLoss
from utils import compute_global_accuracy, fast_hist, per_class_iu, poly_lr_scheduler
import random
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.cuda import amp
import torch
from model.build_BiSeNet import BiSeNet
import os
from torch.utils.data import DataLoader
from dataset.voc import VOCSegmentation as VOC
from dataset.transform import *
import contextlib
from pprint import pprint
import argparse
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings(action="ignore")
sys.path.append(os.getcwd())


# setup the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform(random_crop_size, further_data_aug):
    initial = [
        RandomResizedCrop(random_crop_size, (0.5, 2.0)),
        RandomHorizontalFlip(),
    ]
    added = [
        RandomRotation(5),
        # ColorJitter(brightness=(0.0, 3.0), contrast=(0.0, 3.0),
        #             saturation=(0.0, 3.0), hue=(0.5, 0.5))
    ]
    finalize = [
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    train_transform = Compose(
        initial+added+finalize if further_data_aug else initial+finalize
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
            hist += fast_hist(label.flatten(),
                              prediction.flatten(), args.num_classes)

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
    images, _ = iter(data).next()
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
    losses = {"dice": DiceLoss(), "crossentropy": torch.nn.CrossEntropyLoss(
        ignore_index=255)}
    loss_func = losses[args.loss]
    if args.use_lrScheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            threshold=0.0001,
            min_lr=0,
        )
    max_miou = 0
    step = 0
    # start training
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        # lr = optimizer.param_groups[0]['lr']
        lr = poly_lr_scheduler(optimizer, args.learning_rate,
                               iter=epoch, max_iter=args.num_epochs)
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
            if scaler:
                cm = amp.autocast()
            else:
                cm = dummy_cm()

            with cm:
                optimizer.zero_grad()   
                output, output_sup1, output_sup2 = model(data)
                loss1 = loss_func(output, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss = loss1 + loss2 + loss3

            # backward
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if args.use_lrScheduler:
                scheduler.step(loss)


            tq.update(args.batch_size)
            tq.set_postfix(loss=f"{loss:.4f}", lr=lr)
            step += 1
            # log the progress
            writer.add_scalar("loss_step", loss, step)
            loss_record.append(loss.item())
            principal_loss_record.append(loss1.item())

        tq.close()
        loss_train_mean = np.mean(loss_record)
        pri_train_mean = np.mean(principal_loss_record)
        writer.add_scalar("epoch/loss_epoch_train",
                          float(loss_train_mean), epoch)
        writer.add_scalar("epoch/pri_loss_epoch_train",
                          float(pri_train_mean), epoch)

        if epoch % args.checkpoint_step == 0 or epoch == args.num_epochs:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.save_model_path, f"model_{epoch}_.pth"),
            )

        if (epoch % args.validation_step == 0 or
            epoch == args.num_epochs or
                epoch == 1):
            precision, miou, val_loss = val(
                args, model, dataloader_val, loss_func)
            if miou > max_miou:
                max_miou = miou
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_model_path,
                                 f"best_{args.loss}_loss.pth"),
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
        "--checkpoint_step",
        type=int,
        default=2,
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
    parser.add_argument("--num_workers", type=int,
                        default=4, help="num of workers")
    parser.add_argument(
        "--num_classes", type=int, default=21, help="num of object classes (with void)"
    )
    parser.add_argument(
        "--cuda", type=str, default="0", help="GPU ids used for training"
    )
    parser.add_argument(
        "--use_gpu", type=str2bool, default=True, help="whether to user gpu for training"
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
        type=str2bool,
        default=True,
        help="use automatic mixed precision",
    )
    parser.add_argument(
        "--use_lrScheduler",
        type=str2bool,
        default=False,
        help="Permit the use of lr Scheduler",
    )
    parser.add_argument(
        "--random_crop_size",
        type=int,
        default=310,
        help="Random crop size for data augmentation during training",
    )
    parser.add_argument(
        "--further_data_aug",
        type=str2bool,
        default=False,
        help="Permit use of new data augmentations for training",
    )
    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
        raise ValueError(
            f"optimizer not supported optimizer: {args.optimizer}")

    return optimizer


def main(params):
    # parse the parameters
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args(params)
    print("Training with following arguments:")
    pprint(vars(args), indent=4, compact=True)
    print("Running on: {}".format(device if args.use_gpu else torch.device('cpu')))
    # create dataset and dataloader
    train_path = args.data
    train_transform, val_transform = get_transform(
        args.random_crop_size, args.further_data_aug)

    dataset_train = VOC(train_path, image_set="train",
                        transform=train_transform)
    dataset_val = VOC(train_path, image_set="val", transform=val_transform)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, drop_last=True, )
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size,
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
    print("Training completed.", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))


def fix_seed(seed=44):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    params = [
        "--num_epochs", "30",
        "--batch_size", "32",
        "--learning_rate", "0.005",
        "--context_path", "resnet101",  # set resnet18, resnet50 or resnet101
        "--optimizer", "sgd",
        "--random_crop_size", "320",
        "--data", "/content/root_drive/MyDrive/data" if os.name != 'nt' else
        r"C:\Users\rehma\Google Drive\data",
        "--save_model_path", "/root_drive/MyDrive/models/res101_20_005_sgd",
        "--num_workers", "12",
        "--validation_step", "2",
        "--num_classes", "21",
        "--cuda", "0",
        "--use_gpu", "True",
        "--use_amp", "True",
        "--use_lrScheduler", "False",
        "--further_data_aug", "False",
        # "--pretrained_model_path", "/root_drive/MyDrive/models/res18_20_01_sgd/model.pth"
    ]
    print("started:", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    fix_seed(44)
    main(params)
    # main(sys.argv[1:])
