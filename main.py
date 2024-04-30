import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import time

class MineSelectionModel(nn.Module):
    def __init__(self, num_mines):
        super(MineSelectionModel, self).__init__()
        self.rnn = nn.RNN(num_mines * 2 + 1, 128, num_layers=2, batch_first=True, dropout=0.33)
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_mines)

    def forward(self, x):
        h0 = torch.zeros(2, 1, 128).to(x.device)  # make h0 a 1-D tensor
        out, _ = self.rnn(x.unsqueeze(0), h0)  # add an extra dimension to x and h0
        out = out.squeeze(0)  # remove the extra dimension from out
        out = self.fc1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return torch.softmax(out, dim=0)

class Mine:
    MIN_FACTOR = 0.7
    MAX_FACTOR = 1.3

    def __init__(self, number, true_value):
        self.ordinal_number = number
        self.true_value = true_value
        self.under_threat = False
        self.days_worked = 0
        self.total_extracted_gold = 0
        self.capture_day = -1

    def extract_gold(self):
        self.days_worked += 1
        extracted_today = self.true_value * random.uniform(self.MIN_FACTOR, self.MAX_FACTOR)
        self.total_extracted_gold += extracted_today
        return extracted_today

    def is_under_threat(self):
        return self.under_threat

    def get_days_worked(self):
        return self.days_worked

    def get_total_extracted_gold(self):
        return self.total_extracted_gold

    def get_capture_day(self):
        return self.capture_day

    def get_ordinal_number(self):
        return self.ordinal_number

    def capture(self, day):
        self.under_threat = True
        self.capture_day = day

class Rebels:
    ATTACK_PERIOD = 10
    ATTACK_SUCCESS_PROBABILITY = 0.05

    def __init__(self):
        pass

    def move(self, day, mines):
        if day % self.ATTACK_PERIOD != 0:
            return
        for mine in mines:
            if mine.is_under_threat() or random.random() > self.ATTACK_SUCCESS_PROBABILITY:
                continue
            mine.capture(day)

class SupremeLeader:
    def __init__(self, num_mines, target_earnings):
        self.target_earnings = target_earnings
        self.total_earned = 0
        self.mines = [Mine(i, random.randint(1000, 4000)) for i in range(num_mines)]
        self.model = MineSelectionModel(num_mines)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.5, weight_decay=0.05)
        self.criterion = nn.CrossEntropyLoss()

    def select_mine(self, day):
        mine_values = []
        for mine in self.mines:
            if mine.is_under_threat():
                mine_values.extend([0, 0])
                continue
            if mine.days_worked == 0:
                mine_values.extend([0, 0])
            else:
                mine_values.extend([mine.total_extracted_gold, mine.days_worked])

        mine_values = torch.tensor(mine_values)
        mine_values = (mine_values - mine_values.min()) / (mine_values.max() - mine_values.min())
        mine_values = mine_values.unsqueeze(0)
        day_tensor = torch.tensor([day], dtype=torch.float32).unsqueeze(0)
        mine_values = torch.cat([mine_values, day_tensor], dim=1)

        mine_probabilities = self.model(mine_values)
        mine_probabilities = mine_probabilities.squeeze().detach().numpy()

        if not np.all(np.isfinite(mine_probabilities)):
            mine_probabilities = np.ones_like(mine_probabilities) / len(mine_probabilities)

        selected_mine = random.choices(self.mines, mine_probabilities)[0]
        return selected_mine, mine_probabilities

    def exploit_mines(self, day):
        selected_mine, mine_probabilities = self.select_mine(day)
        if selected_mine:
            gold_extracted = selected_mine.extract_gold()
            self.total_earned += gold_extracted
        return selected_mine, mine_probabilities

    def play_game(self):
        rebels = Rebels()
        game_data = []
        for day in range(365):
            selected_mine, mine_probabilities = self.exploit_mines(day)
            mine_features = []
            for mine in self.mines:
                if mine.is_under_threat():
                    mine_features.extend([0, 0])
                    continue
                if mine.days_worked == 0:
                    mine_features.extend([0, 0])
                else:
                    mine_features.extend([mine.total_extracted_gold, mine.days_worked])
            mine_features.append(day)
            game_data.append((mine_features, selected_mine.get_ordinal_number()))
        reward = (self.total_earned / self.target_earnings) * 2 if self.total_earned > self.target_earnings else -1
        return game_data, reward

    def train_model(self, epochs, train_data, train_labels):
        train_data = train_data.float()
        train_labels = train_labels.float()
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(train_data)
            _, reward = self.play_game()  # Unpack the results of play_game
            loss = -reward * self.criterion(outputs, train_labels.long())
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
        torch.save(self.model.state_dict(), 'rnnstate.pth')

    def load_model(self):
        self.model.load_state_dict(torch.load('rnnstate.pth'))
        self.model.eval()

# end line

amount_of_games = 100  # Increase this number

# Vary the number of mines and target earnings
trainingmode = int(input("Enter 1 for training mode, 0 for testing mode: "))
leader = SupremeLeader(num_mines=10, target_earnings=1000000)
num_mines=10
target_earnings=1000000
if trainingmode == 1:
    print("---------------BEGIN DATA COLLECTION--------------")
    #wait
    time.sleep(3)
    train_data = []
    train_labels = []


    leader = SupremeLeader(num_mines=num_mines, target_earnings=target_earnings)
    for _ in range(amount_of_games):
        game_data, reward = leader.play_game()
        for data, label in game_data:
            train_data.append(data)
            train_labels.append(label * min(1, abs(reward)))  # Weigh the labels by the reward
        print(f'Number of mines: {num_mines}, Target earnings: {target_earnings}, Total earned: {leader.total_earned}')

    # Convert lists to tensors
    train_data = [torch.tensor(data) for data in train_data]
    train_data = pad_sequence(train_data, batch_first=True)
    train_labels = torch.tensor(train_labels)
    print("---------------BEGIN TRAINING--------------")
    #wait
    time.sleep(3)
    # Train the model
    leader.train_model(10, train_data, train_labels)
else:
    if trainingmode == 0:
        leader.load_model()
        num_wins = 0
        num_losses = 0
        amount_of_games = 100
        for _ in range(amount_of_games):
            leader = SupremeLeader(num_mines=num_mines, target_earnings=target_earnings)
            print("game #", _, "/", amount_of_games)
            leader.play_game()
            if leader.total_earned > leader.target_earnings:
                num_wins += 1
            else:
                num_losses += 1

        win_rate = num_wins / (num_wins + num_losses) * 100
        loss_rate = num_losses / (num_wins + num_losses) * 100

        print("<---------------------------------------->")
        print(f"Win rate (executed {num_wins + num_losses} times): {win_rate:.2f}%")
        print(f"Loss rate (executed {num_wins + num_losses} times): {loss_rate:.2f}%")




# Collect the data from the game and use it for training
# # Create an instance of SupremeLeader
#
# trainingmode = int(input("Enter 1 for training mode, 0 for testing mode: "))
#
# if trainingmode == 1:
#     train_data = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
#                                [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]])
#     leader.train_model(20, train_data, torch.tensor([0, 1]))
# else:
# Load the trained model

