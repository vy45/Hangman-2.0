class HangmanMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct_guesses = 0
        self.total_guesses = 0
        self.games_won = 0
        self.games_played = 0
        
    def update(self, prediction, target, game_won):
        self.correct_guesses += (prediction == target).sum().item()
        self.total_guesses += len(prediction)
        self.games_won += game_won
        self.games_played += 1
        
    def compute(self):
        return {
            'guess_accuracy': self.correct_guesses / max(1, self.total_guesses),
            'win_rate': self.games_won / max(1, self.games_played)
        } 