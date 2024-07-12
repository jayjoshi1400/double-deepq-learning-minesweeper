# double-deepq-learning-minesweeper
A project on implementing the reinforcement learning technique of double deep-Q learning on minesweeper

Model Architecture: As this is a DDQN, both the target network and online network share the same architecture. The input size is equal to the state size, which in the case of training was 36, since the training board is 6 by 6 and there are 36 cells. Similarly, the output size is also 36. There are two hidden layers with 128 neurons each. After the feature extraction layer, the network branches into two:

Advantage: Estimates the advantage of each possible action.
Value: Estimates overall value.
The forward pass has the following steps:

The input state is normalized by dividing by 8.
A safe softmax function is used. This softmax ensures that the values which cannot be taken are masked.
The advantage and value functions are computed using the safe softmax function. The final Q-values are calculated by combining the value and advantage sub-networks.
The model selects an action either by choosing the highest Q-value or a random valid action to promote exploration. A replay buffer is required for DDQNs to break up the correlation the model may learn between each subsequent time step. The replay buffer is set to 100,000 and the batch size to 4096.

Training Process:

Initially, a training loop is set. This loop runs for 20,000 epochs. The epsilon-greedy strategy is used for choosing actions.
When an action is executed, the agent receives the next state, reward, and whether the game has reached a terminal state.
Each transition is stored in the buffer.
After each addition of 4096 episodes, a batch is sampled from the buffer.
For each sample in the batch, the online model predicts the Q-value. Target Q-values are predicted by the target network and temporal difference loss is calculated using the Bellman equation.
The online model’s weights are updated for each sample. However, the target model’s weights are updated after certain periods.
Actions may also be selected randomly. This randomness is reduced over time and reliance on the model is increased.
