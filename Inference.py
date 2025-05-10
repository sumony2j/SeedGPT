
## Load required modules

import argparse
import torch
import json
from SeedGPT import SeedGPT,Block,MultiHead,Head,FFNet

## Command line arguments

parser = argparse.ArgumentParser(description="Inference")

parser.add_argument("--model_path",type=str,default="./SeedGPT.pt",help="Enter the path of the model")
parser.add_argument("--tokenizer_path",type=str,default="./tokenizer.json",help="Enter the path of the tokenizer")
parser.add_argument("--input",type=str,default="Hello",help="Input for the LLM")
parser.add_argument("--max_token",type=int,default=1000,help="Enter the number of tokens need to be generated")
parser.add_argument("--output_file",type=str,default="./llm_output.txt",help="Enter the path of the output file")
parser.add_argument("--show",action='store_true',help="Show the generated output on stdout (Default: True)")

args = parser.parse_args()

## Load the tokenizer & process

with open(args.tokenizer_path,"r")  as f:
    tok = json.load(f)

ctoi = {k:int(v) for k,v in tok["ctoi"].items()}
itoc = {int(k):v for k,v in tok["itoc"].items()}

def encoder(s):
    return [ctoi[c] for c in s]
def decoder(l):
    return ''.join([itoc[i] for i in l])

## Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(args.model_path,map_location=device)
model.to(device)
model.eval()

## Preprocess the input

llm_input = torch.tensor(encoder(args.input),dtype=torch.long).to(device=device)

# Fix shape
if llm_input.dim() == 1:
        llm_input = llm_input.unsqueeze(0)

output = model.generate(llm_input,args.max_token)

## Print/Save the generated tokens 

decoded_text = decoder(output[0].tolist())

if(args.show):
    print(decoded_text)

print(f"\nThe output is saved in file {args.output_file}\n")
with open(args.output_file,"w") as f:
    f.write(decoded_text)

