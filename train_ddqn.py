import time
import torch
import numpy as np
import sys
sys.path.insert(1,"./Models")
import torch.nn as nn
from ddqn import DoubleDQN, ReplayBuffer
from game import MineSweeper
from numpy import float32
from torch.autograd import Variable
from multiprocessing import Process
from torch import FloatTensor,LongTensor


class GameAgent():

    def __init__(self,width,height,bomb_no,render_flag):

        self.width = width
        self.height = height
        self.bomb_no = bomb_no
        self.box_count = width*height
        self.game_environment = MineSweeper(self.width,self.height,self.bomb_no)
        self.current_model = DoubleDQN(self.box_count,self.box_count)
        self.target_model = DoubleDQN(self.box_count,self.box_count)
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.current_model.parameters(),lr=0.003,weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=2000,gamma=0.95)
        self.target_model.load_state_dict(self.current_model.state_dict())
        self.buffer = ReplayBuffer(100000)
        self.gamma = 0.99
        self.render_flag = render_flag
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.90
        self.reward_threshold = 0.12
        self.reward_step = 0.01
        self.batch_size = 4096
        self.tau = 5e-5
        self.log = open("./Logs/ddqn_log.txt",'w')

        if(self.render_flag):
            self.Render = Render(self.game_environment.state)

    
    def retrieve_models(self,number):
        path = "./pre-trained/ddqn_dnn"+str(number)+".pth"
        weights = torch.load(path)
        self.current_model.load_state_dict(weights['current_state_dict'])
        self.target_model.load_state_dict(weights['target_state_dict'])
        self.optimizer.load_state_dict(weights['optimizer_state_dict'])
        self.current_model.epsilon = weights['epsilon']


    def determine_action(self,state,visibility_mask):
        state = state.flatten()
        visibility_mask = visibility_mask.flatten()
        action = self.current_model.select_action(state,visibility_mask)
        return action

    def execute_step(self,action):
        i = int(action/self.width)
        j = action%self.width
        if(self.render_flag):
            self.Render.state = self.game_environment.state
            self.Render.draw()
            self.Render.bugfix()
        next_state,terminal,reward = self.game_environment.choose(i,j)
        next_fog = 1-self.game_environment.fog
        return next_state,terminal,reward,next_fog

    def update_epsilon(self,average_reward):
        if(average_reward>self.reward_threshold):
            self.current_model.epsilon = max(self.epsilon_min,self.current_model.epsilon*self.epsilon_decay)
            self.reward_threshold += self.reward_step
    
    def calculate_td_loss(self):
        state,action,visibility_mask,reward,next_state,next_mask,terminal = self.buffer.retrieve(self.batch_size)

        state      = Variable(FloatTensor(float32(state)))
        visibility_mask      = Variable(FloatTensor(float32(visibility_mask)))
        next_state = FloatTensor(float32(next_state))
        action     = LongTensor(float32(action))
        next_mask      = FloatTensor(float32(next_mask))
        reward     = FloatTensor(reward)
        done       = FloatTensor(terminal)


        current_q_value      = self.current_model(state,visibility_mask)
        next_q_values = self.target_model(next_state,next_mask)

        q_value          = current_q_value.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        loss_print = loss.item()    

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        for target_param, local_param in zip(self.target_model.parameters(), self.current_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        return loss_print

    def save_checkpoints(self,batch_no):
        path = f"./pre-trained/ddqn_dnn{iteration_number}.pth"
        torch.save({
            'epoch': batch_no,
            'current_state_dict': self.current_model.state_dict(),
            'target_state_dict' : self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon':self.current_model.epsilon
        }, path)

    def save_logs(self,batch_no,avg_reward,loss,wins):
        res = [
                    str(batch_no),
                    "\tAvg Reward: ",
                    str(avg_reward),
                    "\t Loss: ",
                    str(loss),
                    "\t Wins: ", 
                    str(wins),
                    "\t Epsilon: ",
                    str(self.current_model.epsilon)
        ]
        log_line = " ".join(res)
        print(log_line)
        self.log.write(log_line+"\n")
        self.log.flush()


def main():

    game_controller = GameAgent(6,6,6,False)
    state = game_controller.game_environment.state
    epochs = 20000
    save_every = 2000
    count = 0
    running_reward = 0 
    batch_no = 0
    wins=0
    total=0
    
    while(batch_no<epochs):
        
        visibility_mask = 1- game_controller.game_environment.fog
        action = game_controller.determine_action(state,visibility_mask)
        next_state,terminal,reward,_ = game_controller.execute_step(action)
        game_controller.buffer.push(state.flatten(),action,visibility_mask.flatten(),reward,next_state.flatten(),(1-game_controller.game_environment.fog).flatten(),terminal)
        state = next_state
        count+=1
        running_reward+=reward

        if(terminal):
            if(reward==1):
                wins+=1
            game_controller.game_environment.reset()
            state = game_controller.game_environment.state
            visibility_mask = game_controller.game_environment.fog
            total+=1

        if(count==game_controller.batch_size):
            game_controller.current_model.train()
            loss = game_controller.calculate_td_loss()
            game_controller.current_model.eval()

            batch_no+=1
            avg_reward = running_reward/game_controller.batch_size
            wins = wins*100/total
            game_controller.save_logs(batch_no,avg_reward,loss,wins)

            game_controller.update_epsilon(avg_reward)
            running_reward=0
            count=0
            wins=0
            total=0

            if(batch_no%save_every==0):
                game_controller.save_checkpoints(batch_no)

main()