import argparse
import numpy as np
from sample_factory.utils.utils import str2bool
from sf_examples.nethack.utils.nle_tokenizer.tokenizer import NLE_TOKENIZER, NLE_TOKENIZER_TOK_2_STR, NLE_TOKENIZER_TUPLE_2_TOK
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import wandb

import nle.dataset as nld


class VFNet(nn.Module):
    def __init__(self, k_dim, bagged_trajectory):
        super(VFNet, self).__init__()

        self.k_dim = k_dim
        self.bagged_trajectory = bagged_trajectory

        max_token = np.max(list(NLE_TOKENIZER.values()))
        self.token_embed = nn.Embedding(max_token + 1, self.k_dim, padding_idx=0)

        self.fc = nn.Linear(self.k_dim, 1)

    def forward(self, msg_tokens, mask):
        # Embed each token, mapping masked tokens to the origin
        msg_embed = self.token_embed(msg_tokens.long()) * mask[..., None]

        # Sum all embeddings in a message (BoW assumption) and FC to get message value
        msg_embed = msg_embed.sum(axis=2)

        if self.bagged_trajectory:
            traj_embed = msg_embed.mean(axis=1)
            v = self.fc(traj_embed)[:, 0]
        else:
            # Calculate a instantaneous r at each timestep
            r = self.fc(msg_embed)[:, :, 0]

            # Sum values over each trajectory (trajectory decomposition assumption)
            v = r.mean(axis=1)

        return v


def tokenize_top_line(message, max_token_length):
    # Remove punctuation and NULL characters
    message = message[message != 0]
    message = message[message != ord('!')]
    message = message[message != ord(',')]

    # Split by spaces
    message = np.split(message, np.where(message == ord(' '))[0])
    message_tok_ind = np.array([NLE_TOKENIZER_TUPLE_2_TOK[tuple(x)] for x in message])[:max_token_length]
    
    # Pad with zeros
    message_tok_ind_padded = np.zeros(max_token_length, dtype=np.int32)
    message_tok_ind_padded[:len(message_tok_ind)] = message_tok_ind

    # Replace spaces with null token
    message_tok_ind_padded[message_tok_ind_padded == NLE_TOKENIZER[' ']] = 0

    return message_tok_ind_padded


def train(args, model, device, train_loader, optimizer, epoch):
    print("2")
    model.train()
    for batch_idx, mb in enumerate(train_loader):
        print("3")
        # (batch_size, seq_length, tty_row, tty_col)
        toplines = mb["tty_chars"][:, :, 0, :]
        target = torch.tensor(mb["game_data"][:, 0] / 10000., dtype=torch.float32)

        message_tokens = np.zeros((args.batch_size, args.seq_length, args.max_token_length), dtype=np.int32)
        token_mask = np.ones((args.batch_size, args.seq_length, args.max_token_length), dtype=bool)

        for i in range(args.batch_size):
            print(i)
            done = False
            for j in range(args.seq_length):
                message_tokens[i, j] = tokenize_top_line(toplines[i, j], args.max_token_length)
                if mb["done"][i, j]:
                    done = True

                token_mask[i, j] &= not done
                token_mask[i, j] &= ~(message_tokens[i, j] == 0)

        data, mask, target = torch.tensor(message_tokens).to(device), torch.tensor(token_mask).to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, mask)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

        wandb.log({
            "train_loss": loss.cpu().detach().numpy().mean()
        })

        return

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--seq-length', type=int, default=50000)
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')

    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--max-token-length', type=int, default=16)
    parser.add_argument('--bagged-trajectory', action="store_true")

    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)


    model = VFNet(args.embedding_dim, args.bagged_trajectory).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    wandb.init(
        name="VF"
    )

    for epoch in range(1, args.epochs + 1):
        print("epoch", epoch)
        print("0")
        train_loader = nld.TtyrecDataset("nld-aa-taster-v0", batch_size=args.batch_size, seq_length=args.seq_length)
        print("1")
        train(args, model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader)
        scheduler.step()

    print("ending.")

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()