
'''
B -> BATCH
T/CONTEXT/context -> SEQUENCE
C -> CHANNEL
V -> VOCAB SIZE
H -> HEAD SIZE
E -> EMBEDDING SIZE
'''

## Load required libraries

## ----------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import argparse
import json
## ----------------------------------------------------------------


## Get batch 

## ----------------------------------------------------------------
def get_batch(Dataset,BATCH_SIZE,CONTEXT):
    data = train_data if Dataset == "train" else test_data
    idx = torch.randint(len(data)-CONTEXT,size=(BATCH_SIZE,)) # (B)
    x = torch.stack([data[i:i+CONTEXT] for i in idx]) # (B,T)
    y = torch.stack([data[i+1:i+CONTEXT+1] for i in idx]) # (B,T) 
    return x.to(device),y.to(device)
## ----------------------------------------------------------------

## Head class

## ----------------------------------------------------------------
class Head(torch.nn.Module):
    def __init__(self, head_size, emb_size, context):
        super().__init__()
        self.Q = torch.nn.Linear(emb_size,head_size,bias=False) 
        self.K = torch.nn.Linear(emb_size,head_size,bias=False)
        self.V = torch.nn.Linear(emb_size,head_size,bias=False)
        self.register_buffer("tril",torch.tril(torch.ones(context,context)))  
        self.dropout = nn.Dropout(0.2) 
    def forward(self,x):
        B,T,C = x.shape
        q = self.Q(x) # B,T,H
        k = self.K(x) # B,T,H
        v = self.V(x) # B,T,H
        wei = q @ k.transpose(-2,-1) * C ** -0.05  # (B,T,H) * (B,H,T) = (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # (B,T,T) * (T,T) = (B,T,T)
        wei = torch.softmax(wei,dim=-1) #(B,T,T)
        wei = self.dropout(wei)
        out = wei @ v #(B,T,T) * (B,T,H) = (B,T,H)
        return out # (B,T,H)
## ----------------------------------------------------------------

## MultiHead class

## ----------------------------------------------------------------
class MultiHead(nn.Module):
    def __init__(self, head_size,emb_size,num_head,context):
        super().__init__()
        self.multihead = nn.ModuleList([Head(head_size,emb_size,context) for _ in range(num_head)])
        self.proj = nn.Linear(emb_size,emb_size)
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        out =  torch.cat([h(x) for h in self.multihead],dim=-1) # (B,T,num_head *H) = (B,T,E)
        out = self.dropout(self.proj(out)) # (B,T,E)
        return out
## ----------------------------------------------------------------

## FeedForwardNet class

## ----------------------------------------------------------------
class FFNet(nn.Module):
    def __init__(self, emb_size):
        super().__init__() 
        self.ff = nn.Sequential(
            nn.Linear(emb_size,emb_size*4),
            nn.ReLU(),
            nn.Linear(emb_size*4,emb_size),
            nn.Dropout(0.2)
        )
    def forward(self,x):
        return self.ff(x)
## ----------------------------------------------------------------
    
## Block

## ----------------------------------------------------------------
class Block(nn.Module):
    def __init__(self,emb_size,context,num_head):
        super().__init__()
        self.head_size = emb_size//num_head
        self.head = MultiHead(self.head_size,emb_size,num_head,context)
        self.ff = FFNet(emb_size)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)
    def forward(self,x):
        x = x + self.head(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x # (B,T,E)
## ----------------------------------------------------------------

## ----------------------------------------------------------------
@torch.no_grad
def est_loss(eval_iter):
    out = {}
    model.eval()
    for data in ["train","eval"]:
        losses = []
        for i in range(eval_iter):           
            x,y = get_batch(data,BATCH_SIZE,CONTEXT)
            _,loss = model(x,y)
            losses.append(loss.mean().item())
        avg = sum(losses)/len(losses)
        out[data] = avg
    model.train()
    return out
## ----------------------------------------------------------------
           
## Model

## ----------------------------------------------------------------
class SeedGPT(torch.nn.Module):
    def __init__(self,context,vocab_size,n_layers,emb_size,num_head):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size,emb_size)
        self.pos_tok = torch.nn.Embedding(context,emb_size)
        self.blocks = nn.Sequential(*[Block(emb_size,context,num_head) for _ in range(n_layers)],
                                    nn.LayerNorm(emb_size))
        self.lm = nn.Linear(emb_size,vocab_size)
        self.context = context     
    
    def forward(self,ix,target=None):
        B,T = ix.shape
        tok_emb = self.emb(ix) # (B,T,E)
        tok_pos = self.pos_tok(torch.arange(T,device=ix.device)) # (T,E)
        ix = tok_emb + tok_pos # (B,T,E)
        ix = self.blocks(ix) # (B,T,E)
        logits = self.lm(ix) # (B,T,V)
         
        B,T,C = logits.shape # C = V
        if target is not None:
            logits = logits.view(B*T,C)
            targets = target.view(B*T)
            loss = F.cross_entropy(logits,targets)
        else:
            loss = None
        return logits,loss
    
    def generate(self,ix,max_token):
        for i in range(max_token):
            x = ix[:,-self.context:]
            logit,loss = self(x)
            logit = logit[:,-1,:]
            probs = F.softmax(logit,dim=-1)
            next_ix = torch.multinomial(probs,num_samples=1)
            ix = torch.cat((ix,next_ix),dim=-1)
        return ix
## ----------------------------------------------------------------


## Training the model

def train(iteration,eval_itr):
    opt = torch.optim.AdamW(model.parameters(),lr=lr)
    model.train()
    for i in range(iteration):
        if (i % eval_itr == 0) or (eval_itr == i-1):
            tmp_eval = est_loss(eval_iter=eval_itr)
            print(f"[{i:<5}] Training Loss: {tmp_eval['train']:<20.6f} Evaluation Loss: {tmp_eval['eval']:<20.6f}")
        tr,ts = get_batch("train",BATCH_SIZE,CONTEXT)
        output,loss = model(tr,ts)
        opt.zero_grad(set_to_none=True)
        loss = loss.mean()
        loss.backward()
        opt.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train your LLM model -> SeedGPT")

    parser.add_argument("--batch_size",type=int,help="No. of batches required for training",default=16)
    parser.add_argument("--iteration",type=int,help="Training Epoches",default=1)
    parser.add_argument("--dataset",type=str,
                        help="Dataset path on which MircoGPT is needed to be trained (Currently ont .txt type dataset)",default="./Data.txt")
    parser.add_argument("--context",type=int,help="Size of context required for LLM",default=8)
    parser.add_argument("--emb_size",type=int,help="Size of embedding layer",default=100)
    parser.add_argument("--n_layers",type=int,help="No. of layers/blocks of attention",default=6)
    parser.add_argument("--lr",type=float,help="Learning rate of the training process",default=1e-5)
    parser.add_argument("--n_head",type=int,help="No. of heads",default=4)
    parser.add_argument("--eval_itr",type=int,help="No. of iteration for evaluation",default=10)

    args = parser.parse_args()

    ## ----------------------------------------------------------------

    ## HyperParameter 

    ## ----------------------------------------------------------------
    BATCH_SIZE = args.batch_size
    CONTEXT = args.context
    N_LAYER = args.n_layers
    EMB_SIZE = args.emb_size
    N_HEAD = args.n_head
    lr = args.lr
    iteration = args.iteration
    dataset = args.dataset
    eval_itr = args.eval_itr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## ----------------------------------------------------------------

    ## Load Dataset

    ## ----------------------------------------------------------------
    with open(dataset, "r") as f:
        text = f.read()
    ## ----------------------------------------------------------------

    ## Preprocess the data

    ## ----------------------------------------------------------------
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    ## ----------------------------------------------------------------

    ## Encode & Decoder

    ## ----------------------------------------------------------------
    ctoi = {c:i for i,c in enumerate(vocab)}
    itoc = {i:c for i,c in enumerate(vocab)}

    def encoder(s):
        return [ctoi[c] for c in s]
    def decoder(l):
        return ''.join([itoc[i] for i in l])
    ## ----------------------------------------------------------------

    ## Save the tokenizer

    with open("tokenizer.json","w") as f:
        json.dump({"ctoi":ctoi,"itoc":itoc},f)

    ## Encode the dataset

    ## ----------------------------------------------------------------
    encoded_data = encoder(text)
    data = torch.tensor(encoded_data,dtype=torch.long)
    ## ----------------------------------------------------------------

    ## Data Split

    ## ----------------------------------------------------------------
    train_data = data[:int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):]
    ## ----------------------------------------------------------------

    model = SeedGPT(CONTEXT,vocab_size,N_LAYER,EMB_SIZE,N_HEAD)
    model = nn.DataParallel(model)
    model.to(device=device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")
    print(f"Model Summary : {model}")

    train(iteration,eval_itr)

    ## Save the model
    #torch.save(model.module,"SeedGPT.pt")





    
    
        

        
     

