import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time


import os

class MineSelectionModel(nn.Module):
    def __init__(self, num_mines):
        super(MineSelectionModel, self).__init__()
        self.fc1 = nn.Linear(num_mines * 2 + 1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_mines)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=0)
        return x


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

    def train_model(self, epochs, train_data, train_labels):
        train_data = train_data.float()  # Convert train_data to Float
        train_labels = train_labels.float()  # Convert train_labels to Float
        for epoch in range(epochs):
            self.model.train()  # Set the model to training mode
            self.optimizer.zero_grad()  # Reset gradients

            # Ensure the input data has the correct shape
            train_data = train_data.view(-1, len(self.mines) * 2 + 1)

            outputs = self.model(train_data)  # Forward pass
            reward = self.play_game()  # Get reward or penalty
            loss = -reward * self.criterion(outputs, train_labels.long())  # Multiply loss by reward or penalty
            loss.backward()  # Backward pass
            self.optimizer.step()  # Update weights

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

        # Save the model's state after training
        torch.save(self.model.state_dict(), 'model_state.pth')


    def load_model(self):
        # Load the model's state from the file
        self.model.load_state_dict(torch.load('model_state.pth'))
        self.model.eval()  # Set the model to evaluation mode
    def __init__(self, num_mines, target_earnings):
        self.target_earnings = target_earnings
        self.total_earned = 0
        self.mines = [Mine(i, random.randint(1000, 4000)) for i in range(num_mines)]
        self.mine_expected_values = {}
        self.model = MineSelectionModel(num_mines)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.5)
        self.criterion = nn.CrossEntropyLoss()


    def select_mine(self, day):
        mine_values = []
        for mine in self.mines:
            if mine.is_under_threat():
                mine_values.extend([0, 0])  # Append two default values if mine is under threat
                continue
            if mine.days_worked == 0:
                mine_values.extend([0, 0])  # Append two default values if mine.days_worked is zero
            else:
                mine_values.extend([mine.total_extracted_gold, mine.days_worked])

        mine_values = torch.tensor(mine_values)
        mine_values = (mine_values - mine_values.min()) / (mine_values.max() - mine_values.min())
        mine_values = mine_values.unsqueeze(0)
        day_tensor = torch.tensor([day], dtype=torch.float32).unsqueeze(0)
        mine_values = torch.cat([mine_values, day_tensor], dim=1)

        mine_probabilities = self.model(mine_values)
        mine_probabilities = mine_probabilities.squeeze().detach().numpy()

        # Check if all weights are finite
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
        correct_predictions = 0
        for day in range(365):
            selected_mine, _ = self.exploit_mines(day)
            rebels.move(day, self.mines)
            best_mine = max(self.mines, key=lambda mine: mine.total_extracted_gold)
            if selected_mine == best_mine:
                correct_predictions += 1
        accuracy = correct_predictions / 365
        if self.total_earned > self.target_earnings:
            return 2  # Return reward
        else:
            return -1  # Return penalty




# end line
num_wins = 0
num_losses = 0
# Create an instance of SupremeLeader
leader = SupremeLeader(num_mines=10, target_earnings=1000000)
trainingmode=int(input("Enter 1 for training mode, 0 for testing mode: "))

if trainingmode==1:
    train_data = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                               [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]])
    leader.train_model(100, train_data, torch.tensor([0, 1]))
else:
    # Load the trained model
    leader.load_model()

    amount_of_games = 1000
    for _ in range(amount_of_games):
        leader = SupremeLeader(10, 1000000)
        #clear console
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

