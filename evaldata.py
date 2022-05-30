import metrique
import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from  pytorch_msssim import MS_SSIM
from costumDataset import Kaiset,Kaiset2,feriel
import sys
from torchvision.utils import save_image
#chooses what model to train
if config.MODEL == "ResUnet":
    from resUnet import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import localtime
import os
if not os.path.exists("evaluation"):
    os.mkdir("evaluation")
writer=SummaryWriter("train{}-{}".format(localtime().tm_mon,localtime().tm_mday))
torch.backends.cudnn.benchmark = True
if not os.path.exists(sys.argv[9]):
    os.mkdir(sys.argv[9])
if not os.path.exists(sys.argv[9]+"/image"):
    os.mkdir(sys.argv[9]+"/image")
if not os.path.exists("original"):
    os.mkdir("original")

def test_fn(
    disc, gen, loader, metric, bce, epoch=0
):
    loop = tqdm(loader, leave=True)
    disc.eval()
    gen.eval()
    with torch.no_grad():
     resultat=[]
     resultat2 = []
     resultat3 = []
     resultat4 = []
     resultat5 = []
     for idx, (x, z) in enumerate(loop):
        x = x.to(config.DEVICE)

        # Train Discriminator


        y_fake = gen(x)
        y_fake=y_fake*0.5+0.5
        save_image(y_fake,r"C:\Users\SALIM\PycharmProjects\evalproject\feriel dataset\predict2\\"+z[0])


    return True
def main():
    #instancing the models
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    #print(disc)
    gen = Generator(init_weight=config.INIT_WEIGHTS).to(config.DEVICE)
    #print(gen)
    #instancing the optims
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    schedulergen = torch.optim.lr_scheduler.ExponentialLR(opt_gen , gamma=0.1)
    schedulerdisc = torch.optim.lr_scheduler.ExponentialLR(opt_disc, gamma=0.1)
    #instancing the Loss-functions
    BCE = nn.BCEWithLogitsLoss()

    L1_LOSS = nn.L1Loss()

    ssim = MS_SSIM(data_range=1, size_average=True, channel=3, win_size=11)
    mse=nn.MSELoss()
    #if true loads the checkpoit in the ./
    if sys.argv[6]!="none":
        load_checkpoint(
            sys.argv[6], gen, opt_gen, config.LEARNING_RATE,
        )

    #training data loading

    test_dataset = feriel(path=sys.argv[1])
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(sys.argv[4]),
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    #enabling MultiPrecision Mode, the optimise performance
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    #evauation data loading
    best=10000000
    resultat=1
    for epoch in range(1):
        x=test_fn(disc, gen, test_loader,  [L1_LOSS,ssim,mse], BCE, epoch=epoch)

if __name__ == "__main__":
    main()