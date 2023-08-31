import numpy as np
import torch
import torch.nn as nn
from model import VAE, vae_loss
from tqdm import tqdm
import cv2
from utils import AHE, collate
import torch.nn.functional as F
import lazy_dataset
from sacred import Experiment
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device set to : {device}')

ex = Experiment('VAE', save_git_info=False)
sw = SummaryWriter()

@ex.config
def defaults():
    batch_size = 16
    lr = 0.0001
    steps_per_eval = 1000
    max_steps = 150_000
    latent_size = 256
#load the files
def load_img(example):
    img = cv2.imread(example['image_path'])
    example['image'] = img.astype(np.float32) / 255.0
    return example

@ex.capture
def prepare_dataset(dataset,batch_size):
    if isinstance(dataset,list):
        dataset = lazy_dataset.new(dataset)
    dataset = dataset.map(load_img)
    dataset = dataset.shuffle()
    dataset = dataset.batch(batch_size=batch_size, drop_last=True)
    dataset = dataset.map(collate)
    return dataset

path = 'ckpt_latest.pth'
@ex.automain
def main(batch_size,lr, steps_per_eval, max_steps):
    #model hyperparamters
    #per the LSGAN paper, beta1 os set to 0.5

    db = AHE()
    t_ds = db.get_dataset('training_set')
    v_ds = db.get_dataset('validation_set')
    steps = 0
    model = VAE().to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(),lr=lr)

    for epoch in range(10000):
        epoch_loss = 0
        train_ds = prepare_dataset(t_ds, batch_size=batch_size)
        valid_ds = prepare_dataset(v_ds, batch_size=1)
        for index,batch in enumerate(tqdm(train_ds)):
            optim.zero_grad()
            images = batch['image']
            images = torch.tensor(np.array(images)).to(device).permute(0,3,1,2) #(bs, h, w, 3) --> (bs, 3 ,h, w)
            generated, mu, var = model(images)
            loss = vae_loss(images, generated, mu, var)
            epoch_loss += loss.item()
            loss.backward()
            optim.step()


            if steps % steps_per_eval == 0:
                model.eval()
                with torch.no_grad():
                    for batch in tqdm(valid_ds[0:1]):
                        images = batch['image']
                        images = torch.tensor(np.array(images)).to(device).permute(0,3,1,2)
                        generated, mu, var= model(images)
                        loss = vae_loss(images,generated,mu, var)
                


                    print(f'validation loss after {steps} batches: {loss.item()}')
                    sw.add_scalar("validation/generator_loss",loss,steps)
                    sw.add_images("validation/generated_images", generated,steps)
                    sw.add_images("validation/real_images", images, steps)


                
                torch.save({
                    'steps': steps,
                    'model':model.state_dict() ,
                    'generator_optimizer': optim.state_dict(),
                    }, path)
                
            if steps ==max_steps:
                print('maximum steps reached....stopping training loop')
                break   
            steps +=1
            model.train
        sw.add_scalar("training/generator_loss",epoch_loss/len(train_ds),epoch)
        print(f'epoch {epoch+1} loss: {epoch_loss / len(train_ds)}')
