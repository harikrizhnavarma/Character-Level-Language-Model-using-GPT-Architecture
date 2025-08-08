from GPT_block import GPT
from GPT_dataPrep import DataPrep
import torch
import torch.optim as optim
from hyper_parameters import LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, BLOCK_SIZE
from statistics import mean
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device recogonised as {device}. Available GPU:{torch.cuda.get_device_name(0)}")


obj = DataPrep(dataset_name = 'input.txt')

# instantiating the model
model = GPT(vocab_size = obj.vocab_size).to(device)

# create an adam optimizer
optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE)

#training loop
best_model = model.state_dict() # saving the initial model state

loss_type = dict()
train_losses = []
val_losses = []
least_loss = None
for epoch in range(NUM_EPOCHS):
    for split_type in ['train', 'validate']:
        
        if split_type == 'train':

            # retrieve input and target data for training.
            input_data, target_data = obj.get_batch(split = split_type, block_size = BLOCK_SIZE, batch_size = BATCH_SIZE)
            
            model.train()  # set model to train mode for training loop
        
            #forward passing
            _, loss = model.forward(input = input_data, target = target_data)

            # saving the best model and its loss
            if least_loss is None:
                least_loss = loss
            elif loss < least_loss:
                least_loss = loss

            optimizer.zero_grad(set_to_none = True) # reset the gradients to zero.
            loss.backward() # backward propogation 
            optimizer.step() # adjusting the weights

            train_losses.append(loss.item())
        
        elif split_type == 'validate':

            input_data, target_data = obj.get_batch(split = split_type, block_size = BLOCK_SIZE, batch_size = BATCH_SIZE)            

            model.eval()

            _, loss = model.forward(input = input_data, target = target_data)

            if least_loss is None:
                least_loss = loss
            elif loss < least_loss:
                least_loss = loss
                best_model = model.state_dict() # saving the model with least loss value
            
            val_losses.append(loss.item())

        loss_type[split_type] = least_loss

    if epoch % 500 == 0:
        print(f"Epoch {epoch} - Mean Training Loss:{mean(train_losses):.3f} | Mean Evaluation Loss:{mean(val_losses):.3f}")
    if epoch == NUM_EPOCHS - 1:
        last_model = model.state_dict()
    

print(f"Least Training Loss: {loss_type['train']:.3f} and Validation Loss: {loss_type['validate']:.3f}")

#load the best model
model.load_state_dict(best_model)
torch.save(model.state_dict(), 'GPT.pth')