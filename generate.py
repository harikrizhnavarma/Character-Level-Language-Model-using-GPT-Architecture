import torch
from GPT_block import GPT
from GPT_dataPrep import DataPrep
from hyper_parameters import MAX_NEW_TOKENS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dp = (DataPrep(dataset_name = 'input.txt'))

model = GPT(vocab_size = dp.vocab_size).to(device)

model_name = 'GPT.pth'
model.load_state_dict(torch.load(model_name))

model.eval()

# providing a context for the GPT model to generate upon.
context = 'You may speak citizen'
encoded_context = dp.encode(context)

# provide the encoded context to the generate function as a tensor shape 2D.
output = model.generate(input = torch.tensor([encoded_context], dtype = torch.long, device = device), max_new_tokens = MAX_NEW_TOKENS)

# decoding the tokens 
decoded_output = dp.decode(output[0].tolist())

# save decoded text to a file
with open("GPT_output_test.txt", 'w') as f:
    f.write(decoded_output)

print("Generation Complete ! File saved to local directory.")