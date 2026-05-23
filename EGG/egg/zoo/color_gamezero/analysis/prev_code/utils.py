import os
import torch
import torch.nn as nn
import pathlib
from torch.utils.data import DataLoader, Subset, random_split
from egg.zoo.color_gamezero.models import NewReceiver, NewSender
from egg.zoo.color_gamezero.datasets import UniformDataset


def cal_accuracy(output, labels):
    # pred = output.argmax(dim=1, keepdim=True)
    pred = output.argmax(dim=1)
    # print(f"pred: {pred.shape}, labels: {labels.shape}")
    correct = pred.eq(labels.view_as(pred)).sum().item()
    return correct / len(labels)


# def initialize_weights(m):
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.constant_(m.weight, 0.01)
#         if m.bias is not None:
#             torch.nn.init.constant_(m.bias, 0)


# def worker_init_fn(worker_id):
#     seed = torch.initial_seed() % (2**32)
#     np.random.seed(seed)
#     random.seed(seed)


def load_data(data_path):

    # Set the seed for reproducibility
    # generator = torch.Generator().manual_seed(seed)

    sl_data_set = UniformDataset(data_path, rf=False)
    rf_data_set = UniformDataset(data_path, rf=True)

    half_len = len(sl_data_set) // 2
    first_dataset = Subset(sl_data_set, range(half_len))
    second_dataset = Subset(rf_data_set, range(half_len, len(rf_data_set)))
    # first_dataset = Subset(sl_data_set, range(half_len, len(sl_data_set)))
    # second_dataset = Subset(rf_data_set, range(half_len))

    sl_train_size = int(0.8 * len(first_dataset))
    sl_valid_size = len(first_dataset) - sl_train_size

    rf_train_size = int(0.5 * len(second_dataset))
    rf_valid_size = len(second_dataset) - rf_train_size

    sl_train_set,sl_valid_set = random_split(first_dataset, [sl_train_size, sl_valid_size])
    rf_train_set,rf_valid_set = random_split(second_dataset, [rf_train_size, rf_valid_size])



    sl_train_loader = DataLoader(sl_train_set, batch_size=1024, shuffle=True, drop_last=True)
    sl_valid_loader = DataLoader(sl_valid_set, batch_size=1024, shuffle=False, drop_last=True)

    rf_train_loader = DataLoader(rf_train_set, batch_size=1024, shuffle=True, drop_last=True)
    rf_valid_loader = DataLoader(rf_valid_set, batch_size=1024, shuffle=False, drop_last=True)

    # sl_train_loader = DataLoader(sl_train_set, batch_size=1024, shuffle=True, drop_last=False, num_workers=1, worker_init_fn=worker_init_fn)
    # sl_valid_loader = DataLoader(sl_valid_set, batch_size=1024, shuffle=False, drop_last=False, num_workers=1, worker_init_fn=worker_init_fn)

    # rf_train_loader = DataLoader(rf_train_set, batch_size=1024, shuffle=True, drop_last=False, num_workers=1)
    # rf_valid_loader = DataLoader(rf_valid_set, batch_size=1024, shuffle=False, drop_last=False, num_workers=1)
    return sl_train_loader, sl_valid_loader, rf_train_loader, rf_valid_loader, rf_data_set.color_num, rf_data_set.id_to_color



def create_sender(color_num, embed_dim, hidden_dim, dropout, if_context, lr, device): # num_classes = color_num
    model = NewSender(embed_dim, hidden_dim, color_num, dropout, if_context).to(device)
    # model.apply(initialize_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer



def create_receiver(color_num, embed_dim, hidden_dim, dropout, lr, device):
    model = NewReceiver(embed_dim, hidden_dim, color_num, dropout).to(device)
    # model.apply(initialize_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer



def train_sender(color_num, embed_dim, hidden_dim, dropout, if_context, lr, train_loader, valid_loader, device, n_epochs, id_to_color, seed):

    if if_context:
        training_log_dir = pathlib.Path("training_log_context")
        dump_dir = pathlib.Path("dump_context")
    else:
        training_log_dir = pathlib.Path("training_log")
        dump_dir = pathlib.Path("dump")

    training_log_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths
    log_file_path = training_log_dir / f"log_spk_seed{seed}.txt"  # Log file for accuracy
    spk_folder = dump_dir / f"msg_spk_seed{seed}"
    os.makedirs(spk_folder, exist_ok=True)

    model, optimizer = create_sender(color_num, embed_dim, hidden_dim, dropout, if_context, lr, device)
    
    with open(log_file_path, "w") as log_file:  # Open log file for writing
        for epoch in range(n_epochs):
            model.train()
            train_outs = []
            train_ids = []
            for i, (sender_input, position, receiver_input, color_ids) in enumerate(train_loader):
                sender_input = sender_input.to(device)
                color_ids = color_ids.to(device)
                optimizer.zero_grad()
                output = model(sender_input)
                loss = nn.CrossEntropyLoss()(output, color_ids)
                loss.backward()
                optimizer.step()
                train_outs.append(output)
                train_ids.append(color_ids)
            acc = cal_accuracy(torch.cat(train_outs), torch.cat(train_ids))
            log_file.write(f"Epoch {epoch}, train acc: {acc}\n")  # Log training accuracy
            print(f"Epoch {epoch}, train acc: {acc}")
            
            model.eval()
            with torch.no_grad():
                valid_outs = []
                valid_ids = []
                with open(os.path.join(spk_folder, f"sender_epoch{epoch}.txt"), "w") as f:
                    for i, (sender_input, position, receiver_input, color_ids) in enumerate(valid_loader):
                        sender_input = sender_input.to(device)
                        color_ids = color_ids.to(device)
                        output = model(sender_input)
                        valid_outs.append(output)
                        valid_ids.append(color_ids)

                        colors = [id_to_color[color_id.item()] for color_id in color_ids]
                        preds = [id_to_color[output[i].argmax().item()] for i in range(len(output))]
                        for i in range(sender_input.shape[0]):
                            f.write(f"{[[int(x) for x in sublist] for sublist in sender_input[i].tolist()]} -> {preds[i]} (label={colors[i]})\n")
                
                acc = cal_accuracy(torch.cat(valid_outs), torch.cat(valid_ids))
                log_file.write(f"Epoch {epoch}, valid acc: {acc}\n")  # Log validation accuracy
                print(f"Epoch {epoch}, valid acc: {acc}")
    return model



def train_receiver(color_num, embed_dim, hidden_dim, dropout, if_context, lr, train_loader, valid_loader, device, n_epochs, id_to_color, seed):

    if if_context:
        training_log_dir = pathlib.Path("training_log_context")
        dump_dir = pathlib.Path("dump_context")
    else:
        training_log_dir = pathlib.Path("training_log")
        dump_dir = pathlib.Path("dump")

    training_log_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    log_file_path = training_log_dir / f"log_lst_seed{seed}.txt"  # Log file for accuracy
    lst_folder = dump_dir / f"msg_lst_seed{seed}"
    os.makedirs(lst_folder, exist_ok=True)
        
    model, optimizer = create_receiver(color_num, embed_dim, hidden_dim, dropout, lr, device)
    
    with open(log_file_path, "w") as log_file:  # Open log file for writing
        for epoch in range(n_epochs):
            model.train()
            train_outs = []
            train_pos = []
            for i, (sender_input, position, receiver_input, color_ids) in enumerate(train_loader):
                receiver_input = receiver_input.to(device)
                color_ids = color_ids.to(device)
                position = position.to(device)
                optimizer.zero_grad()
                output = model(color_ids, receiver_input)
                loss = nn.CrossEntropyLoss()(output, position)
                loss.backward()
                optimizer.step()
                train_outs.append(output)
                train_pos.append(position)
            acc = cal_accuracy(torch.cat(train_outs), torch.cat(train_pos))
            log_file.write(f"Epoch {epoch}, train acc: {acc}\n")  # Log training accuracy
            print(f"Epoch {epoch}, train acc: {acc}")
            
            model.eval()
            with torch.no_grad():
                valid_outs = []
                valid_pos = []
                with open(os.path.join(lst_folder, f"receiver_epoch{epoch}.txt"), "w") as f:
                    for i, (sender_input, position, receiver_input, color_ids) in enumerate(valid_loader):
                        receiver_input = receiver_input.to(device)
                        color_ids = color_ids.to(device)
                        position = position.to(device)
                        output = model(color_ids, receiver_input)
                        valid_outs.append(output)
                        valid_pos.append(position)

                        preds = output.argmax(dim=1)
                        colors = [id_to_color[color_id.item()] for color_id in color_ids]
                        for i in range(receiver_input.shape[0]):
                            f.write(
                                f"{colors[i]} -> {[[int(x) for x in sublist] for sublist in receiver_input[i].tolist()]} -> {preds[i]} (label={position[i]})\n"
                            )

                acc = cal_accuracy(torch.cat(valid_outs), torch.cat(valid_pos))
                log_file.write(f"Epoch {epoch}, valid acc: {acc}\n")  # Log validation accuracy
                print(f"Epoch {epoch}, valid acc: {acc}")
    return model


