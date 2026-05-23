This code implements the signalling game described in [1]. The game proceeds as follows:
 * Sender is shown a target image alongside with one or many distractor images,
 * Sender sends a one-symbol message to Receiver,
 * Receiver obtains Sender's message and all images in random order,
 * Receiver predicts which of the received images is the target one and agents are rewarded if the prediction is correct.

HLS 

id_to_colors: {0: 'mustard', 1: 'cyan', 2: 'maroon', 3: 'lavander', 4: 'medium', 5: 'blood', 6: 'turquoise', 7: 'purple', 8: 'blue', 9: 'grapes', 10: 'caca', 11: 'teal', 12: 'sky', 13: 'grass', 14: 'red', 15: 'seafoam', 16: 'aqua', 17: 'clay', 18: 'barney', 19: 'green', 20: 'concrete', 21: 'pumpkin', 22: 'drab', 23: 'tan', 24: 'neon', 25: 'olive', 26: 'lavender', 27: 'fuchsia', 28: 'gray', 29: 'magenta', 30: 'grape', 31: 'dull', 32: 'peach', 33: 'mint', 34: 'mauve', 35: 'yellow', 36: 'sage', 37: 'brown', 38: 'pink', 39: 'beige', 40: 'gold', 41: 'orange', 42: 'salmon', 43: 'bright', 44: 'seagreen', 45: 'violet', 46: 'khaki', 47: 'rose', 48: 'lime'}
id_to_colors: {0: 'mustard', 1: 'cyan', 2: 'maroon', 3: 'lavander', 4: 'medium', 5: 'blood', 6: 'turquoise', 7: 'purple', 8: 'blue', 9: 'grapes', 10: 'caca', 11: 'teal', 12: 'sky', 13: 'grass', 14: 'red', 15: 'seafoam', 16: 'aqua', 17: 'clay', 18: 'barney', 19: 'green', 20: 'concrete', 21: 'pumpkin', 22: 'drab', 23: 'tan', 24: 'neon', 25: 'olive', 26: 'lavender', 27: 'fuchsia', 28: 'gray', 29: 'magenta', 30: 'grape', 31: 'dull', 32: 'peach', 33: 'mint', 34: 'mauve', 35: 'yellow', 36: 'sage', 37: 'brown', 38: 'pink', 39: 'beige', 40: 'gold', 41: 'orange', 42: 'salmon', 43: 'bright', 44: 'seagreen', 45: 'violet', 46: 'khaki', 47: 'rose', 48: 'lime'}

The game can be launched with the following command (with appropriate path to the data):

# python -m egg.zoo.signal_game.train --root=/private/home/kharitonov/work/egg/data/concepts/
python -m egg.zoo.color_game.train



What SL data look like: 
tar_CIELAB, contents_cleaned
"(47.6965536549228, 42.99169356824062, -9.950603922785351)",['pink']
"(47.69658198075641, 65.74381642552774, 7.81018440872554)",['magenta']
"(47.69739906298078, 30.35812573486846, -76.78703087754148)",['blue']
"(47.69761228099255, 25.855856750681415, -25.277208774020245)","['purple', 'pink']"
Normalize: a range of [0, 1] or a mean of 0 and standard deviation of 1




# def train_sender(color_num, embed_dim, hidden_dim, dropout, if_context, lr, train_loader, valid_loader, device, n_epochs, id_to_color, seed):
#     # Create folders based on seed
#     spk_folder = f"./msg_spk_seed{seed}"
#     os.makedirs(spk_folder, exist_ok=True)
    
#     model, optimizer = create_sender(color_num, embed_dim, hidden_dim, dropout, if_context, lr, device)
#     for epoch in range(n_epochs):
#         model.train()
#         train_outs = []
#         train_ids = []
#         for i, (sender_input, position, receiver_input, color_ids) in enumerate(train_loader):
#             sender_input = sender_input.to(device)
#             color_ids = color_ids.to(device)
#             optimizer.zero_grad()
#             output = model(sender_input)
#             loss = nn.CrossEntropyLoss()(output, color_ids)
#             loss.backward()
#             optimizer.step()
#             train_outs.append(output)
#             train_ids.append(color_ids)
#         acc = cal_accuracy(torch.cat(train_outs), torch.cat(train_ids))
#         print(f"Epoch {epoch}, train acc: {acc}")
        
#         model.eval()
#         with torch.no_grad():
#             valid_outs = []
#             valid_ids = []
#             with open(os.path.join(spk_folder, f"sender_epoch{epoch}.txt"), "w") as f:
#                 for i, (sender_input, position, receiver_input, color_ids) in enumerate(valid_loader):
#                     sender_input = sender_input.to(device)
#                     color_ids = color_ids.to(device)
#                     output = model(sender_input)
#                     valid_outs.append(output)
#                     valid_ids.append(color_ids)
                    
#                     colors = [id_to_color[color_id.item()] for color_id in color_ids]
#                     preds = [id_to_color[output[i].argmax().item()] for i in range(len(output))]
#                     for i in range(sender_input.shape[0]):
#                         f.write(f"{sender_input[i]} -> {preds[i]} (label={colors[i]})\n")
            
#             acc = cal_accuracy(torch.cat(valid_outs), torch.cat(valid_ids))
#             print(f"Epoch {epoch}, valid acc: {acc}")
#     return model

# def train_receiver(color_num, embed_dim, hidden_dim, dropout, lr, train_loader, valid_loader, device, n_epochs, id_to_color, seed):
#     # Create folders based on seed
#     lst_folder = f"./msg_lst_seed{seed}"
#     os.makedirs(lst_folder, exist_ok=True)
    
#     model, optimizer = create_receiver(color_num, embed_dim, hidden_dim, dropout, lr, device)
#     for epoch in range(n_epochs):
#         model.train()
#         train_outs = []
#         train_pos = []
#         for i, (sender_input, position, receiver_input, color_ids) in enumerate(train_loader):
#             receiver_input = receiver_input.to(device)
#             color_ids = color_ids.to(device)
#             position = position.to(device)
#             optimizer.zero_grad()
#             output = model(color_ids, receiver_input)
#             loss = nn.CrossEntropyLoss()(output, position)
#             loss.backward()
#             optimizer.step()
#             train_outs.append(output)
#             train_pos.append(position)
#         acc = cal_accuracy(torch.cat(train_outs), torch.cat(train_pos))
#         print(f"Epoch {epoch}, train acc: {acc}")

#         model.eval()
#         with torch.no_grad():
#             valid_outs = []
#             valid_pos = []
#             with open(os.path.join(lst_folder, f"receiver_epoch{epoch}.txt"), "w") as f:
#                 for i, (sender_input, position, receiver_input, color_ids) in enumerate(valid_loader):
#                     receiver_input = receiver_input.to(device)
#                     color_ids = color_ids.to(device)
#                     position = position.to(device)
#                     output = model(color_ids, receiver_input)
#                     valid_outs.append(output)
#                     valid_pos.append(position)
                    
#                     preds = output.argmax(dim=1)
#                     colors = [id_to_color[color_id.item()] for color_id in color_ids]
#                     for i in range(receiver_input.shape[0]):
#                         f.write(
#                             f"{colors[i]} -> {receiver_input[i]} -> {preds[i]} (label={position[i]})\n"
#                         )
#             acc = cal_accuracy(torch.cat(valid_outs), torch.cat(valid_pos))
#             print(f"Epoch {epoch}, valid acc: {acc}")
#     return model



# def train_sender(color_num, embed_dim, hidden_dim, dropout, if_context, lr, train_loader, valid_loader, device, n_epochs, id_to_color):
#     model, optimizer = create_sender(color_num, embed_dim, hidden_dim, dropout, if_context, lr, device)
#     # n_epochs = 2
#     for epoch in range(n_epochs):
#         model.train()
#         train_outs = []
#         train_ids = []
#         for i, (sender_input, position, receiver_input, color_ids) in enumerate(train_loader):
#             sender_input = sender_input.to(device)
#             color_ids = color_ids.to(device)
#             optimizer.zero_grad()
#             output = model(sender_input)
#             loss = nn.CrossEntropyLoss()(output, color_ids)
#             loss.backward()
#             optimizer.step()
#             train_outs.append(output)
#             train_ids.append(color_ids)
#         acc = cal_accuracy(torch.cat(train_outs), torch.cat(train_ids))
#         print(f"Epoch {epoch}, train acc: {acc}")
        
#         model.eval()
#         with torch.no_grad():
#             valid_outs = []
#             valid_ids = []
#             # Open a new file for each epoch
#             with open(f"./msg_spk/sender_epoch{epoch}.txt", "w") as f:
#                 for i, (sender_input, position, receiver_input, color_ids) in enumerate(valid_loader):
#                     sender_input = sender_input.to(device)
#                     color_ids = color_ids.to(device)
#                     output = model(sender_input)
#                     valid_outs.append(output)
#                     valid_ids.append(color_ids)
                    
#                     # Save results for this epoch
#                     colors = [id_to_color[color_id.item()] for color_id in color_ids]
#                     preds = [id_to_color[output[i].argmax().item()] for i in range(len(output))]
#                     for i in range(sender_input.shape[0]):
#                         f.write(f"{sender_input[i]} -> {preds[i]} (label={colors[i]})\n")
            
#             acc = cal_accuracy(torch.cat(valid_outs), torch.cat(valid_ids))
#             print(f"Epoch {epoch}, valid acc: {acc}")
#     return model



# def train_receiver(color_num, embed_dim, hidden_dim, dropout, lr, train_loader, valid_loader, device, n_epochs, id_to_color):
#     model, optimizer = create_receiver(color_num, embed_dim, hidden_dim, dropout, lr, device)
#     # n_epochs = 2
#     for epoch in range(n_epochs):
#         model.train()
#         train_outs = []
#         train_pos = []
#         for i, (sender_input, position, receiver_input, color_ids) in enumerate(train_loader):
#             receiver_input = receiver_input.to(device)
#             color_ids = color_ids.to(device)
#             position = position.to(device)
#             optimizer.zero_grad()
#             output = model(color_ids, receiver_input)
#             loss = nn.CrossEntropyLoss()(output, position)
#             loss.backward()
#             optimizer.step()
#             train_outs.append(output)
#             train_pos.append(position)
#         acc = cal_accuracy(torch.cat(train_outs), torch.cat(train_pos))
#         print(f"Epoch {epoch}, train acc: {acc}")

#         model.eval()
#         with torch.no_grad():
#             valid_outs = []
#             valid_pos = []
#             # Open a new file for each epoch
#             with open(f"./msg_lst/receiver_epoch{epoch}.txt", "w") as f:
#                 for i, (sender_input, position, receiver_input, color_ids) in enumerate(valid_loader):
#                     receiver_input = receiver_input.to(device)
#                     color_ids = color_ids.to(device)
#                     position = position.to(device)
#                     output = model(color_ids, receiver_input)
#                     valid_outs.append(output)
#                     valid_pos.append(position)
                    
#                     # Save results for this epoch
#                     preds = output.argmax(dim=1)
#                     colors = [id_to_color[color_id.item()] for color_id in color_ids]
#                     for i in range(receiver_input.shape[0]):
#                         f.write(
#                             f"{colors[i]} -> {receiver_input[i]} -> {preds[i]} (label={position[i]})\n"
#                         )
#             acc = cal_accuracy(torch.cat(valid_outs), torch.cat(valid_pos))
#             print(f"Epoch {epoch}, valid acc: {acc}")
#     return model





# class NewSender(nn.Module):
#     def __init__(self, embed_dim, hidden_dim, num_classes, dropout=0.2, if_context=True):
#         super(NewSender, self).__init__()
#         self.if_context = if_context
#         # 定义层
#         self.hidden1 = nn.Sequential(
#                             nn.Linear(embed_dim, hidden_dim),
#                             nn.BatchNorm1d(hidden_dim),
#                             nn.ReLU(),
#                             nn.Dropout(p=dropout))
#         self.dist1_layer = nn.Sequential(
#                             nn.Linear(embed_dim, hidden_dim),
#                             nn.BatchNorm1d(hidden_dim),
#                             nn.ReLU(),
#                             nn.Dropout(p=dropout))
        
#         self.dist2_layer = nn.Sequential(
#                             nn.Linear(embed_dim, hidden_dim),
#                             nn.BatchNorm1d(hidden_dim),
#                             nn.ReLU(),
#                             nn.Dropout(p=dropout))
#         self.joint_layer = nn.Sequential(
#                             nn.Linear(hidden_dim * 3, hidden_dim),
#                             nn.BatchNorm1d(hidden_dim),
#                             nn.ReLU(),
#                             nn.Dropout(p=dropout))
#         self.classifier = nn.Linear(hidden_dim, num_classes)

#     def forward(self, sender_input, _aux_input=None):
#         x = sender_input[:, 0, :]
#         dist1 = sender_input[:, 1, :]
#         dist2 = sender_input[:, 2, :]
        
#         # Process main input
#         h = self.hidden1(x)
        
#         # Always process context layers but keep parameters frozen when if_context=False
#         with torch.no_grad() if not self.if_context else torch.enable_grad():
#             h1 = self.dist1_layer(dist1)
#             h2 = self.dist2_layer(dist2)
#             h_joint = torch.cat((h, h1, h2), dim=1)
#             h = self.joint_layer(h_joint)
        
#         # Classification layer
#         out = self.classifier(h)
#         log_probs = F.log_softmax(out, dim=1)
#         return log_probs

#     def set_context(self, if_context):
#         """Update the context condition."""
#         self.if_context = if_context



