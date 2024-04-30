import random
import math
import tensorflow as tf
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
        self.mine_expected_values = {}

    def select_mine(self, day):
        UCB_MAX = -float('inf')
        selected_mine = None
        total_days_worked = sum(mine.get_days_worked() for mine in self.mines)
        for mine in self.mines:
            if mine.is_under_threat():
                continue
            if mine.get_days_worked() == 0:
                ucb = float('inf')
            else:
                average_reward = mine.get_total_extracted_gold() / mine.get_days_worked()
                exploration_factor = math.sqrt(2 * math.log(total_days_worked) / mine.get_days_worked())
                ucb = average_reward + exploration_factor
            if ucb > UCB_MAX:
                UCB_MAX = ucb
                selected_mine = mine
        return selected_mine

    def exploit_mines(self, day):
        selected_mine = self.select_mine(day)
        if selected_mine:
            gold_extracted = selected_mine.extract_gold()
            self.total_earned += gold_extracted

    def play_game(self):
        rebels = Rebels()
        for day in range(365):
            self.exploit_mines(day)
            rebels.move(day, self.mines)
        if self.total_earned > self.target_earnings:
            print("You won, gg.")
        else:
            print("You lost, skill issue.")
        print(f"Total earnings: ${self.total_earned:.0f}")

num_wins = 0
num_losses = 0

amount_of_games = int(input("Enter the amount of games to simulate: "))
gamenumber = 1;
for _ in range(amount_of_games):
    leader = SupremeLeader(10, 1000000)
    print("<---------------------------------------->")
    print(f"Game number {gamenumber}")
    leader.play_game()
    gamenumber += 1
    if leader.total_earned > leader.target_earnings:
        num_wins += 1
    else:
        num_losses += 1

win_rate = num_wins / (num_wins + num_losses) * 100
loss_rate = num_losses / (num_wins + num_losses) * 100

print("<---------------------------------------->")
print(f"Win rate (executed {num_wins + num_losses} times): {win_rate:.2f}%")
print(f"Loss rate (executed {num_wins + num_losses} times): {loss_rate:.2f}%")