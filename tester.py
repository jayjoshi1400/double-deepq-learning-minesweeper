import torch
import numpy as np
import sys
sys.path.insert(1,"./Models")
from ddqn import DoubleDQN
# from renderer import Render
from newUI import TkinterRenderer
from game import MineSweeper
import time

class GameEvaluator():
    def __init__(self,render_flag):
        self.model = DoubleDQN(36,36)
        self.render_flag = render_flag
        self.grid_width = 6
        self.grid_height = 6
        self.game_environment = MineSweeper(self.grid_width,self.grid_height,6)
        if(self.render_flag):
            initial_board = self.convert_state(self.game_environment.state)
            self.renderer = TkinterRenderer(initial_board)
        self.retrieve_models(20000)
    
    def convert_state(self, state):
        mapping = {
            # -1: 'F',
            0: ' ',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
        }
        string_state = [[' ' for _ in range(len(state[0]))] for _ in range(len(state))]
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] in mapping:
                    string_state[i][j] = mapping[state[i][j]]
                else:
                    # Handle other cases or raise an error
                    pass
        return string_state


    def determine_action(self,state):
        state = state.flatten()
        visibility_mask = (1-self.game_environment.fog).flatten()
        action = self.model.select_action(state,visibility_mask)
        return action

    def retrieve_models(self,model_number):
        model_path = f"pre-trained/ddqn_dnn{model_number}.pth"
        model_dict = torch.load(model_path)
        self.model.load_state_dict(model_dict['current_state_dict'])
        self.model.epsilon = 0
                    
    def run_game_loop(self):
        self.game_step()  # Start the first game step

    def game_step(self):
        action = self.determine_action(self.game_environment.state)
        next_state, terminal, reward = self.execute_step(action)
        if not terminal:
            # Schedule the next game step
            self.renderer.root.after(500, self.game_step)  # Adjust the delay as needed
        else:
            # Handle game over (reset the game or close the window)
            print("Game Over. Reward:", reward)
            self.game_environment.reset()

    def execute_step(self, chosen_action):
        row_index = int(chosen_action / self.grid_width)
        col_index = chosen_action % self.grid_width
        next_state, terminal, reward = self.game_environment.choose(row_index, col_index)
        
        if self.render_flag:
            new_board = self.convert_state(self.game_environment.state)
            self.renderer.update(new_board)
        return next_state, terminal, reward
    

    
### Tests winrate in "games_no" games
def evaluate_win_rate(games_no):
    tester = GameEvaluator(False)
    state = tester.game_environment.state
    visibility_mask = tester.game_environment.fog
    wins = 0
    game_index=0
    step = 0
    first_loss = 0
    while game_index< games_no:
        step+=1
        action = tester.determine_action(state)
        next_state,terminal,reward = tester.execute_step(action)
        state = next_state
        if(terminal):
            if(step==1 and reward==-1):
                    first_loss+=1
            game_index+=1
            tester.game_environment.reset()
            state = tester.game_environment.state
            if(reward==1):
                wins+=1
            step=0
    
    ### First_loss is subtracted so that the games with first pick as bomb are subtracted
    print("Win Rate: "+str(wins*100/(games_no)))
    print("Win Rate excluding First Loss: "+str(wins*100/(games_no-first_loss)))


def slow_tester():
    tester = GameEvaluator(True)
    state = tester.game_environment.state
    count = 0
    start = time.perf_counter()
    step = 0
    first_loss = 0

    while(True):
        count+=1
        step+=1
        action = tester.determine_action(state)
        next_state,terminal,reward = tester.execute_step(action)
        state = next_state
        print(reward)
        time.sleep(0.5)

        if(terminal):
            if(reward==1):
                print("WIN")
            else:
                
                print("LOSS")
            tester.game_environment.reset()
            step=0
            state = tester.game_environment.state
            break
        
        
def main():
    tester = GameEvaluator(True)
    tester.run_game_loop()
    tester.renderer.root.mainloop()
    # evaluate_win_rate(500)

if __name__ == "__main__":
    main()
