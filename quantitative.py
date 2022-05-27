import metrique
import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from  pytorch_msssim import MS_SSIM
from costumDataset import Kaiset,Kaiset2
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
     for idx, (x, y,z) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator


        y_fake = gen(x)

        D_real = disc(x, y)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = disc(x, y_fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2



        # Train generator

        D_fake = disc(x, y_fake)
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))


        y_fake=y_fake*0.5+0.5
        L1 = metric[0](y_fake, y)

        ssim = metric[1](y_fake.double(), y.double())
        mse =metric[2](y_fake*255, y*255)
        hst=metrique.orientationhist((y_fake[:,0:1,:,:]*255), (y[:,0:1,:,:]*255))
        hst2 = metrique.histogramme((y_fake * 255), (y * 255))
        resultat.append(L1.item()*255)
        resultat2.append(ssim.item())
        resultat3.append(mse.item())
        resultat4.append(hst[0].item())
        resultat5.append(hst2[0].item())
        save_image(y_fake,sys.argv[9]+"/image/"+z[0])
        if sys.argv[10]=="true":
            save_image(y, "original/" + z[0])
        if idx==0:
                hist1=hst[1]
                hist2=hst[2]
                hist_1 = hst2[1]
                hist_2 = hst2[2]
        else:
                hist1=hist1+hst[1]
                hist2 = hist2 + hst[2]
                hist_1 = hist_1 + hst2[1]
                hist_2 = hist_2 + hst2[2]
        if idx % 1 == 0:
            writer.add_scalar("L1 test loss",L1.item()/config.L1_LAMBDA,epoch*(len(loop))+idx)
            writer.add_scalar("D_real test loss", torch.sigmoid(D_real).mean().item(), epoch * (len(loop)) + idx)
            writer.add_scalar("D_fake test loss", torch.sigmoid(D_fake).mean().item(), epoch * (len(loop)) + idx)
            loop.set_postfix(

                L1    =L1.item()*255,
                ssim=ssim.item(),
                mse=mse.item()*255
            )

    return torch.tensor(resultat).mean(),torch.tensor(resultat2).mean(),torch.tensor(resultat3).mean(),torch.tensor(resultat4).mean(),torch.tensor(resultat5).mean(),hist1/idx,hist2/idx,hist_1/idx,hist_2/idx
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
    train_dataset = Kaiset(path=sys.argv[1], Listset=config.DTRAIN_LIST if sys.argv[5]=="0"else config.NTRAIN_LIST)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(sys.argv[4]),
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    test_dataset = Kaiset2(path=sys.argv[1],train=False, Listset=config.DTRAIN_LIST if sys.argv[5]=="0"else config.NTRAIN_LIST)
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
        file=open(sys.argv[9]+"/resultat.txt",'w')
        file.write("L1:"+str(x[0])+"\n")
        file.write("SSIM:"+str(x[1]) + "\n")
        file.write("MSE:"+str(x[2]) + "\n")
        file.write("orientation :"+str(x[3]) + "\n")
        file.write("gray scale :" + str(x[4]) + "\n")
        import matplotlib.pyplot as plt
        plt.plot(range(360),x[5])
        plt.savefig(sys.argv[9]+"/otrue.jpg")
        plt.figure()
        plt.plot(range(360),x[6])
        plt.savefig(sys.argv[9]+"/opred.jpg")
        plt.figure()
        plt.plot(range(255), x[7])
        plt.savefig(sys.argv[9]+"/trueh.jpg")
        plt.figure()
        plt.plot(range(255), x[8])
        plt.savefig(sys.argv[9]+"/predh.jpg")
if __name__ == "__main__":
    main()