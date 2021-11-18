import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.functional import F

from tqdm import tqdm

from utils import abs, bin, sign


class BaseNN(torch.nn.Module):
    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load(self, model_id: str = "default", device=None):
        load_kwargs = {"map_location": device} if device is not None else {}

        try:
            print("Loading from ", f"model/model_{model_id}.pth")
            self.load_state_dict(torch.load(f"model/model_{model_id}.pth", **load_kwargs))

            print("Loading optimizer from ", f"model/optimizer_{model_id}.pth")
            self.optimizer.load_state_dict(torch.load(f"model/optimizer_{model_id}.pth", **load_kwargs))
            self.eval()
        except FileNotFoundError as error:
            error.strerror = "There is no model located on"
            raise

    def train_network(self, n_epochs, model_id, train_loader, test_loader, save_every_time=False):
        train_data = []
        test_data = []

        test_data.append(self._test(test_loader))
        best_acc = float("-inf")
        for _ in tqdm(range(1, n_epochs + 1)):
            train_result = self._train_epoch(train_loader)
            test_result = self._test(test_loader)

            train_data.append(train_result)
            test_data.append(test_result)

            tqdm.write(f"ACC: {(100. * test_result[1]):.3f}% | Loss: {test_result[0]}")
            if test_result[1] > best_acc:
                best_acc = test_result[1]
                tqdm.write(f"Saving model because ACC increased to {(100. * best_acc):.3f}%")
                self._save(f"{model_id}_{(1000. * best_acc):.0f}")
            elif save_every_time:
                tqdm.write("Saving model even without ACC increasing")
                self._save(f"{model_id}_{(1000. * test_result[1]):.0f}")

        return (test_data, train_data)

    def _compute_loss(self, output, target, backward=True):
        raise NotImplementedError("You need to implement this in a subclass")

    def _compute_correct(self, output, target):
        raise NotImplementedError("You need to implement this in a subclass")

    def _test(self, test_loader):
        self.eval()
        loss = 0
        correct = 0

        with torch.no_grad():
            for input in test_loader:
                data, target = input[0]

                output = self(data)

                loss += self._compute_loss(output, target, backward=False).item() / len(target)

                correct += self._compute_correct(output, target)

        loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset)

        return loss, acc

    def _train_epoch(self, train_loader):
        sum_loss = 0
        sum_correct = 0

        self.train()
        with tqdm(total=len(train_loader.dataset)) as pbar:
            for input in train_loader:
                data, target = input[0]

                self.optimizer.zero_grad()

                output = self(data)
                loss = self._compute_loss(output, target)

                if hasattr(self, "clip"):
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
                self.optimizer.step()

                sum_loss += loss.item() / len(target)
                sum_correct += self._compute_correct(output, target)
                pbar.update(len(data))

        loss = sum_loss / len(train_loader.dataset)
        acc = sum_correct / len(train_loader.dataset)

        return loss, acc

    def decode(self, llrs, iters):
        for _ in range(iters):
            sign_llrs = sign(llrs)
            bin_sign_llrs = bin(sign_llrs)
            syndrome = torch.matmul(torch.tensor(bin_sign_llrs), self.H.T) % 2
            abs_llrs = abs(llrs)

            input = torch.cat((syndrome, torch.tensor(abs_llrs)))

            self.optimizer.zero_grad()
            output = self(input.unsqueeze(0))

            # Multiply the output by the input LLRs, because the output means "flip this bit" or not
            llrs = output[0] * llrs
        return llrs

    def _save(self, model_id: str = "default"):
        # Create the model folders if it not exists
        Path("model/").mkdir(parents=True, exist_ok=True)

        current_time = int(time.time())
        torch.save(self.state_dict(), f"model/model_{current_time}_{model_id}.pth")
        torch.save(self.optimizer.state_dict(), f"model/optimizer_{current_time}_{model_id}.pth")


class SyndromeBasedNetwork(BaseNN):
    def __init__(
        self, H, num_inputs, num_outputs, clip=1, learning_rate=1e-3, hidden_multiplier=6, hidden_layers=9, device=None
    ):
        """
        num_inputs (integer): Number of inputs which will be received.
            This refers to the size of the encoded message + the concated syndrome.
        num_outputs (integer): Number of values on the output layer.
            This refers to the size of the encoded message.
        clip (float): Value to clip the gradient to.
        learning_rate (float): LR used on the Adam optimizer
        hidden_multiplier (int): the multiplier to compute the number of nodes on the hidden layers
        hidden_layers (int): the number of hidden layers besides the input and output ones
        device(device): device to move the tensors to
        """
        super(SyndromeBasedNetwork, self).__init__()
        self.H = H
        self.device = device
        self.num_inputs = num_inputs
        self.hidden_multiplier = hidden_multiplier
        self.hidden_layers = hidden_layers
        self.clip = clip

        self.input = torch.nn.Linear(num_inputs, self.hidden_multiplier * num_inputs)
        self.output = torch.nn.Linear((self.hidden_multiplier + 1) * num_inputs, num_outputs)

        # Remember to use torch.nn.ModuleList to actually move these layers to the GPU
        self.internal = [
            torch.nn.Linear((self.hidden_multiplier + 1) * num_inputs, self.hidden_multiplier * num_inputs)
            for _ in range(self.hidden_layers)
        ]
        self.internal = torch.nn.ModuleList(self.internal)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input):
        if self.device:
            input = input.to(self.device)

        input = input.float()

        x = self.relu(self.input(input))
        for layer in self.internal:
            x = self.relu(layer(torch.cat((x, input), dim=1)))

        output = self.tanh(self.output(torch.cat((x, input), dim=1)))
        return output

    def _compute_loss(self, output, target, backward=True):
        if self.device:
            target = target.to(self.device)

        # To compute the loss, we need to map the range [-1, 1] to [0, 1]
        # which is as simple as divide by 2, and sum +0.5
        output = (output / 2) + 0.5
        target = (target / 2) + 0.5

        loss = F.binary_cross_entropy(output.float(), target.float())

        if backward:
            loss.backward()

        return loss

    def _compute_correct(self, output, target):
        correct = 0

        for out, tgt in zip(output.data.cpu(), target.data):
            correct += sum(sign(out) == sign(tgt)) / len(out)

        return correct
