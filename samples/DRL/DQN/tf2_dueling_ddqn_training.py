from tf2_dueling_ddqn_agent import Agent
import sys
sys.path.insert(1, '../../../modules')
import drl


agent = Agent(lr=0.0005, gamma=0.99, n_actions=4, epsilon=1.0,
                  batch_size=64, input_dims=[8])

drl.TDTrain(envName='LunarLander-v2', agent=agent, n_games=10, 
            modelFolderName="dueling_ddqn", 
            plotName="lunarlander-dueling_ddqn.png", save_checkpoint = True, 
            checkpoint_interval = 2)
