from torch_dueling_ddqn_agent import Agent
import sys
sys.path.insert(1, '../../../modules')
import drl
import gen
saveFolderName = "torch_lunarlander_dueling_ddqn_2"
saveDir = gen.makeModelSaveDir(saveFolderName)

agent = Agent(gamma=0.99, epsilon=1.0, lr=5e-4, chkpt_dir=saveDir,
                input_dims=[8], n_actions=4, mem_size=100000, eps_min=0.01,
                batch_size=64, eps_dec=1e-3, replace=100)

drl.TDTrain(envName='LunarLander-v2', agent=agent, n_games=25, modelFolderName=saveFolderName, 
            plotName="torch-lunarlander-dueling_ddqn_2.png", load_checkpoint=False, 
            save_checkpoint=True, checkpoint_interval=5)

