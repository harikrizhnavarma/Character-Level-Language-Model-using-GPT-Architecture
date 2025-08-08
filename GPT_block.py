import torch
import torch.nn as nn
import GPT_dataPrep
from torch.nn import functional as F
from hyper_parameters import NUM_EMBED, BLOCK_SIZE, NUM_HEADS, DROPOUT, NUM_LAYERS
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# instantiate the data preparation script to obj
obj = GPT_dataPrep.DataPrep(dataset_name = 'input.txt') 

""" Define the attention head to communicate between the characters inside a sequence. Characters will not communicate with characters of other batches. """

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.query = nn.Linear(NUM_EMBED, head_size, bias = False) # shape of query is (batch_size x block_size x head_size)
        self.key = nn.Linear(NUM_EMBED, head_size, bias = False) # shape of key is (batch_size x block_size x head_size)
        self.value = nn.Linear(NUM_EMBED, head_size, bias = False) # shape of value is (batch_size x block_size x head_size)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))) # shape because q.k(t) shape is (block_size x block_size)
        
        self.dropout = nn.Dropout(DROPOUT) # drop out layer to prevent overfitting

    def forward(self, input):
        q = self.query(input)
        k = self.key(input)

        B, T, C = input.shape # here inputs will the embeddings done using the token embedding table. transforms (B,T) shaped encoded input to (B, T, C) shape.
        weights = q @ k.transpose(-2, -1) * C**0.5 # shape B T T.
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # shape B T T. masking the upper triangular part of the matrix 
                                                                             # to -inf so that it does not consider future tokens.
        weights = F.softmax(weights, dim = -1) # applying softmax to get the probability of each token in the sequence

        weights = self.dropout(weights)
        v = self.value(input)
        out = weights @ v # dot product of weights and value because we want to get the weighted sum of values

        return out

"""Multi Head attention blocks to help attend the inputs to different aspects instead of only one aspect"""

class MultiheadedAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.multi_heads = nn.ModuleList(Head(head_size) for _ in range(num_heads)) # using class Head, we do multiple self attentions and makes it like a list of module.
        self.proj = nn.Linear(NUM_EMBED, NUM_EMBED)
        
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.multi_heads], dim = -1) # take each heads from the module list, calculate self attention using the input data.
        out = self.dropout(self.proj(out))
        return out

""" Now that the characters have looked at each other, they need to figure out what they want. A simple feedforward layer should do the trick """

class FeedForwardLayer(nn.Module):
    def __init__(self, num_embed):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_embed, 4 * num_embed), # because we want to expand the input to 4 times the size of num_embed as mentioned in the paper 'attention is all you need'.
            nn.ReLU(), #because we want to apply non-linearity to the output of the linear layer.
            nn.Linear(4 * num_embed, num_embed), # and then reduce it back to num_embed size. This allows the model to learn complex relationships.
            nn.Dropout(DROPOUT) # to prevent overfitting
        )
    
    def forward(self, x):
        out = self.net(x)
        return out

""" We define the mutlihead attention and feedforward layer inside a Block to do the calculation multiple times """

class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.sa = MultiheadedAttention(num_heads = NUM_HEADS, head_size = NUM_EMBED // NUM_HEADS) # multiple self attention heads to calculate the attention weights and apply them to the input.
                                                                                                  # we are splitting the num embeds equally to each heads.
        self.ff = FeedForwardLayer(num_embed = NUM_EMBED)

        self.layernorm1 = nn.LayerNorm(NUM_EMBED) # nn.LayerNorm will help the model to learn better and faster by normalizing the input to the self attention layer.
        self.layernorm2 = nn.LayerNorm(NUM_EMBED)
    
    def forward(self, x):
        x = x + self.sa(self.layernorm1(x)) # first we apply layer normalization to the input, then we apply self attention and add the output to the input.
        x = x + self.ff(self.layernorm2(x)) # then we apply layer normalization to the input, then we apply feedforward layer and add the output to the input.
                                            # this is called residual connection, it helps the model to learn better by allowing the gradients to flow through the network without vanishing.
        return x

""" The BigramNN model is defined here which uses the above defined blocks to calculate the logits and loss """

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, NUM_EMBED) # because we want to convert the token indices to embeddings of size NUM_EMBED
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, NUM_EMBED) # because we want to convert the position indices to embeddings of size NUM_EMBED
        self.block = nn.Sequential(*[Block() for _ in range(NUM_LAYERS)])
        self.layernorm = nn.LayerNorm(NUM_EMBED) # to normalize the output of the last block before passing it to the linear layer.
        self.lm_head = nn.Linear(NUM_EMBED, vocab_size) # linear layer transformation to convert the output of self attention head to logits of size vocab_size
    
    def forward(self, input, target = None):

        B, T = input.shape
        token_embed_score = self.token_embedding_table(input)
        position_embed_score = self.position_embedding_table(torch.arange(T, device = input.device)) # shape (T, NUM_EMBED) where T is block size. inp is [0,1,2,3,4,5,6,7]
        
        #calculating the logit values to predict next character by inputting a random character
        input = token_embed_score + position_embed_score # shape (B, T, C)  where C is num_embed.
        input = self.block(input) # shape (B, T, C)
        input = self.layernorm(input) # normalizing the output of the last block before passing it to the linear layer.
        logits = self.lm_head(input) # shape (B, T, vocab_size) in our case vocab_size is 65.

        if target is None: # this if condition is for the generate function because it does not need a target value to generate, and therefore no loss.
            loss = None
        else:
            B,T,C = logits.shape 
            logits = logits.view(B*T, C) # reshape the logits tensor to 2D
            target_data = target.view(B*T) # reshape the target tensor to 1D
            loss = F.cross_entropy(logits, target_data)

        return logits, loss
    
    # defining function to generate next most probable token using the model.
    def generate(self, input, max_new_tokens):
        for _ in range(max_new_tokens):
            inp = input[:, -BLOCK_SIZE:] # keep only the last block_size tokens
            logits, _ = self(inp) # input the new data
            probs = F.softmax(logits[:, -1, :], dim = -1) # indexing only to get the logits value of last token in the sequence since bigram only looks at the last character.
                                                          #Calculate softmax probability.
            next_token = torch.multinomial(probs, num_samples = 1) #retrieve the token index with highest probability
            input = torch.cat((input, next_token), dim = 1) # concat new token with previous token

        return input