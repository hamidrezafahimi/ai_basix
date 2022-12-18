from tf2_dueling_ddqn_agent import Agent
import sys
sys.path.insert(1, '../../../modules')
import drl
import gen


agent = Agent()
agent.load_model(gen.getModelDir("dueling_ddqn"))

drl.justPlayGym('LunarLander-v2', 2, agent)
