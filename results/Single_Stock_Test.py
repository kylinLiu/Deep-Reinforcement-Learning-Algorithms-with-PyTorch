'''
copy from Cart_Pole.py
pip install gym-anytrading
import gym-anytrading

'''
import os
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))

import gym
import gym_anytrading

from agents.actor_critic_agents.A2C import A2C
from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.actor_critic_agents.A3C import A3C
from agents.policy_gradient_agents.PPO import PPO
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from agents.DQN_agents.DQN import DQN
from agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets

config = Config()
config.seed = 1
# symbol="SH113581"
# symbol="SH113575"
# symbol="SH113565"
# symbol="SH113586"
# symbol="SH123056"
# symbol="SH128106"
symbol="SZ123030"
symbol="SH113552"
symbol="SZ128092"
symbol="SH113525"
symbol="SZ127003"
symbol="SH128088"
symbol="SZ123041"
symbol="SZ123029"
symbol="SH123020"
symbol="SH128080"
symbol="SH113536"
symbol="SH113558"
symbol="SH123034"
symbol="SH128059"
symbol="SH127015"
symbol="SH113577"
symbol="SH113585"
symbol="SH113504"
symbol="SH128108"
symbol="SH113572"
symbol="SH113518"
symbol="SZ123018"
symbol="SH113555"
symbol="SZ128043"
symbol="SZ128086"
symbol="SZ123027"
# symbol="SH113554"
# symbol="SH128021"
# symbol="SH113548"
# symbol="SH113578"
# symbol="SH123051"
# symbol="SH128022"
# symbol="SH128079"
# symbol="SH113580"
# symbol="SH113027"
# symbol="SH113571"
# symbol="SH123022"
# symbol="SH113545"
# symbol="SH128075"
# symbol="SH128102"
# symbol="SH113556"
# symbol="SH123037"
# symbol="SH113031"
# symbol="SH128030"
# symbol="SH128089"
# symbol="SH128074"
# symbol="SH113521"
# symbol="SH123026"
# symbol="SH123031"
# symbol="SH128084"
# symbol="SH113547"
# symbol="SH113509"
# symbol="SH113520"
# symbol="SH113514"
# symbol="SH128036"
# symbol="SH123048"
# symbol="SH128045"
# symbol="SH113550"
# symbol="SH113019"
# symbol="SH113028"
# symbol="SH128029"
# symbol="SH128104"
# symbol="SH110042"
# symbol="SH128098"
# symbol="SH128017"
# symbol="SH128115"
# symbol="SH110060"
# symbol="SH113566"
# symbol="SH132021"
# symbol="SH113022"
# symbol="SH128078"
# symbol="SH128039"
# symbol="SH113567"
# symbol="SH113035"
# symbol="SH113541"
# symbol="SH128099"
# symbol="SH123044"
# symbol="SH127005"
# symbol="SH128103"
# symbol="SH128066"
# symbol="SH123025"
# symbol="SH128013"
# symbol="SH113008"
# symbol="SH128053"
# symbol="SH123002"
# symbol="SH113526"
# symbol="SH110066"
# symbol="SH123040"
# symbol="SH113543"
# symbol="SH128114"
# symbol="SH132018"
# symbol="SH123047"
# symbol="SH123038"
# symbol="SH113561"
# symbol="SH128096"
# symbol="SH110058"
# symbol="SZ128105"
# symbol="SH128019"
# symbol="SZ128112"
# symbol="SH123052"
# symbol="SH113553"
# symbol="SH123032"
# config.environment  = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)
# gym_anytrading.register_new('sz.000001')
# gym_anytrading.register_new('sh.600959')
print(symbol)
gym_anytrading.register_new_kzz(symbol)
config.environment = gym.make('kzz-v1')
# config.environment.update_df()
# column_list = ['turn', 'pctChg']
# column_list = ['turn', 'pctChg']
column_list = ["test2"]
# column_list = ["turn", "pctChg", "peTTM", "psTTM", "pcfNcfTTM", "pbMRQ"]
column_list_str = "_".join(column_list)
# config.environment.update_df(fn=None, column_list=column_list)
config.environment.update_df(fn=None, column_list=None)
# config.environment.update_df(fn=lambda df:df.head(100), column_list=column_list)
# config.environment = gym.make("CartPole-v0")
config.num_episodes_to_run = 50
# config.num_episodes_to_run = 450
config.file_to_save_data_results = "results/data_and_graphs/stocks_Results_Data.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/stocks_Results_Graph.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.model_path = r'drive/My Drive/l_gym/Models/%s' % column_list_str
config.save_model = False
config.load_model = True
config.run_test = True
config.run_test_path = r"drive/My Drive/l_gym/data_and_graphs/%s/%s/{}_run_test.png" % (symbol,column_list_str)
# config.run_test_path = r"drive/My Drive/l_gym/data_and_graphs/%s/{}_run_test.png" % column_list_str

try:
    os.makedirs(config.model_path)
    os.makedirs(r"drive/My Drive/l_gym/data_and_graphs/%s/%s" % (symbol,column_list_str))
except:
    pass

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.01,
        "batch_size": 256,
        "buffer_size": 40000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 1,
        "discount_rate": 0.99,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [30, 15],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 0.7,
        "learning_iterations": 1,
        "clip_rewards": False
    },
    "Stochastic_Policy_Search_Agents": {
        "policy_network_type": "Linear",
        "noise_scale_start": 1e-2,
        "noise_scale_min": 1e-3,
        "noise_scale_max": 2.0,
        "noise_scale_growth_factor": 2.0,
        "stochastic_action_decision": False,
        "num_policies": 10,
        "episodes_per_policy": 1,
        "num_policies_to_keep": 5,
        "clip_rewards": False
    },
    "Policy_Gradient_Agents": {
        "learning_rate": 0.05,
        "linear_hidden_units": [20, 20],
        "final_layer_activation": "SOFTMAX",
        "learning_iterations_per_round": 5,
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,
        "episodes_per_learning_round": 4,
        "normalise_rewards": True,
        "gradient_clipping_norm": 7.0,
        "mu": 0.0,  # only required for continuous action games
        "theta": 0.0,  # only required for continuous action games
        "sigma": 0.0,  # only required for continuous action games
        "epsilon_decay_rate_denominator": 1.0,
        "clip_rewards": False
    },

    "Actor_Critic_Agents": {

        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}

if __name__ == "__main__":
    AGENTS = [SAC_Discrete, DDQN, Dueling_DDQN, DQN, DQN_With_Fixed_Q_Targets,
              DDQN_With_Prioritised_Experience_Replay, A2C, PPO, A3C]
    AGENTS = [SAC_Discrete,
              DDQN,
              Dueling_DDQN,
              DQN,
              DQN_With_Fixed_Q_Targets,
              DDQN_With_Prioritised_Experience_Replay,
              # A2C,
              # PPO,
              # A3C
              ]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
