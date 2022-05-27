import torch
from torchvision import transforms

DEVICE = "cpu"
#paths and sets for the data
TRAIN_DIR = r'E:\KAIST-dataset\kaist-cvpr15\images'
DTRAIN_LIST = ["set00" , 'set01' , 'set02' , 'set06' , 'set07',"set08"]
DTEST_LIST = ["set08"]
NTRAIN_LIST = ["set03" , 'set04' , 'set05' , 'set09' , 'set10',"set11"]
NTEST_LIST = ["set11"]
VAL_DIR = r'E:\KAIST-dataset\kaist-cvpr15\images'
#the list of models implemented.
MODEL_LIST = ["ResUnet", "Unet"]
#choosing the model to train.
MODEL = MODEL_LIST[0]
#hyper-parameters
LEARNING_RATE = 2e-4
BATCH_SIZE =1
#the number of images saved by save_some_images
EVAL_BATCH_SIZE = 1
#the ressources allocated to loading the data
NUM_WORKERS = 2
IMAGE_SIZE = 256
#when true, initializes the wieghts with a normal distro with mean 0 and std 0.02 (paper values) if False random Init
INIT_WEIGHTS = True
CHANNELS_IMG = 3
L1_LAMBDA = 10
NUM_EPOCHS = 30
#when true loads the models saved as "disc.pth.tar" and "gen.pth.tar", they need to be in ./
LOAD_MODEL = False
#when true saves a checkpoint every 5 epochs
SAVE_MODEL = True
#the names of the checkpoints
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

#the data pre-processing used.
transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform2 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])

