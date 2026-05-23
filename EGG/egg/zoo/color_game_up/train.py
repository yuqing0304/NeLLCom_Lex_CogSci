# # Copyright (c) Facebook, Inc. and its affiliates.

# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

import random
import argparse
import json
import torch
import os
import egg.core as core
from egg.zoo.color_game.utils_condition import load_data, train_receiver, train_sender
current_dir = os.path.dirname(os.path.abspath(__file__))

import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Training configurations
    parser.add_argument("--mode", type=str, default="rf", choices=["rf", "gs"],
                        help="Training mode: Gumbel-Softmax (gs) or Reinforce (rf). Default: rf.")
    parser.add_argument("--test_color", type=str, required=True)
    parser.add_argument("--spk_hidden_dim", type=int, default=512, help="Hidden layer dimension.") 
    parser.add_argument("--lst_hidden_dim", type=int, default=512, help="Hidden layer dimension") 
    parser.add_argument("--embed_dim", type=int, default=3, help="Input embedding dimension. Default: 3.")
    parser.add_argument("--gs_tau", type=float, default=1.0, help="Gumbel-Softmax temperature. Default: 1.0.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate. Default: 0.2.")
    parser.add_argument("--if_context", action="store_true", help="Set context for sender. Default: False.")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs. Default: 10.")
    parser.add_argument("--n_comm_epochs", type=int, default=20, help="Number of communication training epochs. Default: 10.")
    parser.add_argument("--total_epochs", type=int, default=20, help="Total number of communication training epochs. Default: 20.")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate. Default: 0.01.")  #0.00015
    parser.add_argument("--comm_lr", type=float, default=0.00001, help="Communication learning rate. Default: 0.01.") 
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--n_receivers", type=int, default=1,
    help="Number of receivers in the game. Default: 1.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[111],
        help="List of random seeds, e.g. --seeds 111 222 333"
    )


                          
    opt = core.init(parser)
    return opt



def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    Accuracy loss - non-differetiable hence cannot be used with GS
    """
    acc = (labels == receiver_output).float()

    condition = _aux_input["condition"]

    acc_far = acc[condition == 0]
    acc_close = acc[condition == 1]
    acc_split = acc[condition == 2]

    return -acc, {"acc": acc, "acc_far": acc_far, "acc_close": acc_close, "acc_split": acc_split}


def loss_nll(
    _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
):
    """
    NLL (negative log-likelihood) loss - differentiable and can be used with both GS and Reinforce
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float().mean()
    return nll, {"acc": acc}



def get_game(opts, sender, receiver):
    if opts.mode == "rf":
        sender = core.ReinforceWrapper(sender)
        receiver = core.ReinforceWrapper(receiver)
        game = core.SymbolGameReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=0.00,  ## was 0.15 for ARR, 0.02 for upsampling, 0.15 for Cogsci
            receiver_entropy_coeff=0.00,  ## was 0.15 for ARR, 0.02 for upsampling, 0.15 for Cogsci
        )
    elif opts.mode == "gs":
        sender = core.GumbelSoftmaxWrapper(sender, temperature=opts.gs_tau)
        game = core.SymbolGameGS(sender, receiver, loss_nll)
    else:
        raise RuntimeError(f"Unknown training mode: {opts.mode}")

    return game



def main():
    opts = parse_arguments()
    color_num = 3

    seed_list = opts.seeds
    test_color = opts.test_color


    print("Random seeds for this run:", seed_list)

    for seed in seed_list:
        print(f"\n=== Running experiment with seed {seed} ===")
        set_seed(seed)

        # ------------------------------
        # Load data
        # ------------------------------
        data_path = opts.data_path
        data_type = f"{os.path.splitext(os.path.basename(data_path))[0]}_{opts.n_receivers}lst"

        sl_train_loader, sl_valid_loader, rf_train_loader, rf_valid_loader, color_num, id_to_color = load_data(data_path, test_color)

        print("Start Supervised Learning (SL)...")
        print(f"if_context: {opts.if_context}")

        # ------------------------------
        # Train Sender + multiple Receivers with SL
        # ------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Train receivers (SL)
        receivers = []
        for i in range(opts.n_receivers):
            # r_seed = seed + (i + 1)  # different seed for each receiver
            r_seed = seed + i  
            receiver = train_receiver(
                color_num, opts.embed_dim, opts.lst_hidden_dim, opts.dropout,
                opts.if_context, opts.lr, sl_train_loader, sl_valid_loader,
                device, opts.n_epochs, id_to_color, data_type, test_color, r_seed
            )
            receivers.append(receiver)
        receiver.save_embeddings(filename=f"receiver_emb_supervised_seed{seed}.npy", out_dir=f"{test_color}/outputs_emb_{data_type}")    

        # Train sender (SL)
        sender = train_sender(
            color_num, opts.embed_dim, opts.spk_hidden_dim, opts.dropout,
            opts.if_context, opts.lr, sl_train_loader, sl_valid_loader,
            device, opts.n_epochs, id_to_color, data_type, test_color, seed
        )    
        sender.save_embeddings(filename=f"sender_emb_supervised_seed{seed}.npy", out_dir=f"{test_color}/outputs_emb_{data_type}")
        print(f"Successfully trained 1 Sender + {len(receivers)} Receivers (SL).")


        print("*" * 20)
        print("Start Reinforcement Learning (RL)...")

        # ------------------------------
        # RL training with group communication
        # ------------------------------
        total_epochs = opts.total_epochs   # e.g. 20
        n_receivers = len(receivers)
        epochs_per_receiver = max(1, total_epochs // n_receivers)

        callbacks = [core.ConsoleLogger(as_json=True, print_train_loss=True)]

        # Train sender sequentially with each receiver
        for i, receiver in enumerate(receivers, 1):
            print(f"\n[RL] Training with Receiver {i}/{n_receivers} for {epochs_per_receiver} epochs...")

            # Create game with current sender + receiver
            game = get_game(opts, sender, receiver)

            # Freeze receiver parameters, update only sender
            # for p in receiver.parameters():
            #     p.requires_grad = False
            # optimizer = core.build_optimizer(
            #     filter(lambda p: p.requires_grad, game.sender.parameters()),
            #     rate=opts.comm_lr
            # )
            optimizer = core.build_optimizer(game.parameters(), rate=opts.comm_lr)


            # Train RL
            trainer = core.Trainer(
                game=game,
                optimizer=optimizer,
                train_data=rf_train_loader,
                validation_data=rf_valid_loader,
                callbacks=callbacks,
            )
            trainer.train(
                n_epochs=epochs_per_receiver,
                id_to_color=id_to_color,
                data_type=data_type,
                test_color=test_color,
                current_seed=seed,
                if_context=opts.if_context
            )

            # Update sender for next receiver
            sender = trainer.get_sender()

        print(f"=== Training with seed {seed} completed. ===")

        trainer.save_receiver_embeddings(filename=f"receiver_emb_reinforce_seed{seed}.npy", out_dir=f"{test_color}/outputs_emb_{data_type}")
        trainer.save_sender_embeddings(filename=f"sender_emb_reinforce_seed{seed}.npy", out_dir=f"{test_color}/outputs_emb_{data_type}")

    print("\nAll experiments completed.")


    core.close()

if __name__ == "__main__":
    main()
