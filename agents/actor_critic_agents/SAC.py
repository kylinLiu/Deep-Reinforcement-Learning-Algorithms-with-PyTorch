from agents.Base_Agent import Base_Agent
from utilities.OU_Noise import OU_Noise
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class SAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    """基于2018年论文https://arxiv.org/abs/1812.05905和此github实现https://github.com/pranz24/pytorch-soft-actor-critic的Soft Actor-Critic模型。 这是一种行为批评算法，其中还对代理进行了训练，以使他们的行为及其累积奖励最大化"""
    agent_name = "SAC"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "CONTINUOUS", "Action types must be continuous. Use SAC Discrete instead for discrete actions"
        assert self.config.hyperparameters["Actor"][
                   "final_layer_activation"] != "Softmax", "Final actor layer must not be softmax"
        self.hyperparameters = config.hyperparameters
        self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                             key_to_use="Critic", override_seed=self.config.seed + 1)

        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size * 2,
                                          key_to_use="Actor")

        if self.config.load_model: self.locally_load_policy()
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                            key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                              key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(
                self.device)).item()  # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

    def save_result(self):
        """Saves the result of an episode of the game. Overriding the method in Base Agent that does this because we only
        want to keep track of the results during the evaluation episodes"""
        """保存游戏情节的结果。 覆盖执行此操作的Base Agent中的方法，因为我们只希望在评估情节期间跟踪结果"""
        if self.episode_number == 1 or not self.do_evaluation_iterations:
            self.game_full_episode_scores.extend([self.total_episode_score_so_far])
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
            self.save_max_result_seen()

        elif (self.episode_number - 1) % TRAINING_EPISODES_PER_EVAL_EPISODE == 0:
            self.game_full_episode_scores.extend(
                [self.total_episode_score_so_far for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.rolling_results.extend(
                [np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]) for _ in
                 range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.save_max_result_seen()

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        """重置游戏信息，以便我们准备播放新剧集"""
        Base_Agent.reset_game(self)
        if self.add_extra_noise: self.noise.reset()

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        """在游戏中运行一集，保存经验并在适当时运行学习步骤"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        while not self.done:
            #             print("episode_step_number_val",self.episode_step_number_val)
            self.episode_step_number_val += 1
            #             print("self.state_size/self.action_size:{}/{}".format(self.state_size, self.action_size))
            self.action = self.pick_action(eval_ep)
            #             print("pick_action",self.action)
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
            if not eval_ep:
                #                 print("step save_experience")
                self.save_experience(experience=(self.state, self.action, self.reward, self.next_state, mask))
            self.state = self.next_state
            self.global_step_number += 1
        # print(self.total_episode_score_so_far)
        if eval_ep: self.print_summary_of_latest_evaluation_episode()
        self.episode_number += 1

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        """	Picks an action using one of three methods: 
	      1) Randomly if we haven't passed a certain number of steps,
          2) Using the actor in evaluation mode if eval_ep is True 
	      3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration
	    """
        """
        使用以下三种方法之一选择一个动作：
        1）如果我们没有经过一定数量的步骤，则是随机的，
        2）如果eval_ep为True，则在评估模式下使用actor
        3）如果eval_ep为False，则在训练模式下使用actor。
          评估模式与培训模式之间的差异在于，培训模式需要更多的探索
        """
        if state is None: state = self.state
        if eval_ep:
            action = self.actor_pick_action(state=state, eval=True)
        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            action = self.environment.action_space.sample()
        # print("Picking random action ", action)
        else:
            action = self.actor_pick_action(state=state)
        if self.add_extra_noise:
            action += self.noise.sample()
        return action

    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        """
	    使用actor以两种方式之一选择动作：
        1）如果eval = False，并且我们不在eval模式下，则它将选择已部分随机采样的操作
        2）如果eval = True，则我们选择直接来自网络的操作，因此不涉及任何随机采样“”“
        """
        if state is None: state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1: state = state.unsqueeze(0)
        if eval == False:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        """can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."""
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        """给定状态，产生一个动作，该动作的对数概率和平均动作的tanh"""
        actor_output = self.actor_local(state)
        #         print("actor_output:",actor_output)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        """返回布尔值，指示是否有足够的经验可以学习，是时候让演员和评论家学习"""
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
               self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
            "update_every_n_steps"] == 0

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        """为演员，评论家和温度参数（如果指定）运行演员的学习迭代"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        # print("learn_state_batch",state_batch)
        # print("learn_action_batch",action_batch)
        # print("learn_reward_batch",reward_batch)
        # print("learn_next_state_batch",next_state_batch)
        # print("learn_mask_batch",mask_batch)
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch,
                                                          mask_batch)
        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        if self.automatic_entropy_tuning:
            alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else:
            alpha_loss = None
        self.update_all_parameters(qf1_loss, qf2_loss, policy_loss, alpha_loss)

    def sample_experiences(self):
        return self.memory.sample()

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(torch.cat((next_state_batch, next_state_action), 1))
            qf2_next_target = self.critic_target_2(torch.cat((next_state_batch, next_state_action), 1))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)
        qf1 = self.critic_local(torch.cat((state_batch, action_batch), 1))
        qf2 = self.critic_local_2(torch.cat((state_batch, action_batch), 1))
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(torch.cat((state_batch, action), 1))
        qf2_pi = self.critic_local_2(torch.cat((state_batch, action), 1))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_all_parameters(self, critic_loss_1, critic_loss_2, actor_loss, alpha_loss):
        """Updates the parameters for the actor, both critics and (if specified) the temperature parameter"""
        """更新演员的参数，包括评论者和温度参数（如果指定）"""
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Episode score {} ".format(self.total_episode_score_so_far))
        print("----------------------------")

    def locally_save_policy(self):
        """Saves the policy"""
        """保存策略，待添加"""
        pass
        critic_local_path = "Models/{}_critic_local.pt".format(self.agent_name)
        critic_local_2_path = "Models/{}_critic_local_2.pt".format(self.agent_name)
        actor_local_path = "Models/{}_actor_local.pt".format(self.agent_name)
        torch.save(self.critic_local.state_dict(), critic_local_path)
        torch.save(self.critic_local_2.state_dict(), critic_local_2_path)
        # torch.save(self.critic_target.state_dict(), "Models/{}_critic_target.pt".format(self.agent_name))
        # torch.save(self.critic_target_2.state_dict(), "Models/{}_critic_target_2.pt".format(self.agent_name))
        torch.save(self.actor_local.state_dict(), actor_local_path)

    def locally_load_policy(self):
        import os
        critic_local_path = "Models/{}_critic_local.pt".format(self.agent_name)
        critic_local_2_path = "Models/{}_critic_local_2.pt".format(self.agent_name)
        actor_local_path = "Models/{}_actor_local.pt".format(self.agent_name)
        if os.path.isfile(critic_local_path):
            print("load critic_local_path")
            self.critic_local.load_state_dict(torch.load(critic_local_path))
        if os.path.isifle(critic_local_2_path):
            print("load critic_local_2_path")
            self.critic_local_2.load_state_dict(torch.load(critic_local_2_path))
        if os.path.isfile(actor_local_path):
            print("load actor_local_path")
            self.actor_local.load_state_dict(torch.load(actor_local_path))
