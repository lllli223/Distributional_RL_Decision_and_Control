import json
import numpy as np
import os
import copy
import time


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(value) for value in obj]
    if isinstance(obj, tuple):
        return [_to_serializable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class Trainer():
    def __init__(self,
                 train_env,
                 eval_env,
                 eval_schedule,
                 rl_agent,
                 UPDATE_EVERY=4,
                 learning_starts=2000,
                 target_update_interval=10000,
                #  exploration_fraction=0.5, # Graph IQN
                #  initial_eps=0.5, # Graph IQN
                #  final_eps=0.1, # Graph IQN
                 exploration_fraction=0.25, # IQN
                 initial_eps=0.6, # IQN
                 final_eps=0.05, # IQN
                 imitation=False,
                 il_agent=None,
                 logger=None
                 ):
        
        self.train_env = train_env
        self.eval_env = eval_env
        self.rl_agent = rl_agent
        self.eval_config = []
        self.create_eval_configs(eval_schedule)

        self.UPDATE_EVERY = UPDATE_EVERY
        self.learning_starts = 0 if imitation else learning_starts
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.imitation = imitation
        self.il_agent = il_agent

        if self.imitation:
            assert self.il_agent is not None, "Imitation Learning agent not given!"

        # Check if we're using vectorized environments
        # Import here to avoid circular dependency
        try:
            from parallel_env import SubprocVecEnv
            self.is_vec_env = isinstance(train_env, SubprocVecEnv)
        except ImportError:
            self.is_vec_env = False
        
        if self.is_vec_env:
            self.n_envs = train_env.n_envs
            print(f"Using vectorized environment with {self.n_envs} parallel environments")
        else:
            self.n_envs = 1
            print("Using single environment")

        # Set up logger
        self.logger = logger
        if self.logger is not None:
            log_mode = "vectorized" if self.is_vec_env else "single"
            self.logger.get_logger().info(
                f"Trainer initialized with {self.n_envs} environment(s) in {log_mode} mode"
            )

        # Current time step
        self.current_timestep = 0

        # Learning time step (start counting after learning_starts time step)
        self.learning_timestep = 0

        # Evaluation data
        self.eval_timesteps = []
        self.eval_observations = []
        self.eval_actions = []
        self.eval_trajectories = []
        self.eval_rewards = []
        self.eval_successes = []
        self.eval_times = []
        self.eval_energies = []
        self.eval_relations = []
        # self.eval_obs = []
        # self.eval_objs = []

    def create_eval_configs(self,eval_schedule):
        self.eval_config.clear()

        count = 0
        for i,num_episode in enumerate(eval_schedule["num_episodes"]):
            for _ in range(num_episode): 
                self.eval_env.num_robots = eval_schedule["num_robots"][i]
                self.eval_env.num_cores = eval_schedule["num_cores"][i]
                self.eval_env.num_obs = eval_schedule["num_obstacles"][i]
                self.eval_env.min_start_goal_dis = eval_schedule["min_start_goal_dis"][i]

                self.eval_env.reset()

                # save eval config
                self.eval_config.append(self.eval_env.episode_data())
                count += 1

    def save_eval_config(self,directory):
        file = os.path.join(directory,"eval_configs.json")
        with open(file, "w+") as f:
            json.dump(_to_serializable(self.eval_config), f)

    def learn(self,
              total_timesteps,
              eval_freq,
              eval_log_path,
              verbose=True):
        
        if self.is_vec_env:
            return self._learn_vec_env(total_timesteps, eval_freq, eval_log_path, verbose)
        else:
            return self._learn_single_env(total_timesteps, eval_freq, eval_log_path, verbose)

    def _learn_single_env(self,
              total_timesteps,
              eval_freq,
              eval_log_path,
              verbose=True):
        
        states,_,_ = self.train_env.reset()

        # # Sample CVaR value from (0.0,1.0)
        # cvar = 1 - np.random.uniform(0.0, 1.0)

        # current episode 
        ep_rewards = np.zeros(len(self.train_env.robots))
        ep_deactivated_t = [-1]*len(self.train_env.robots)
        ep_length = 0
        ep_num = 0
        
        # Log episode start
        if self.logger is not None:
            self.logger.log_episode_start(
                env_idx=0,
                episode_num=ep_num,
                num_robots=len(self.train_env.robots),
                timestep=self.current_timestep
            )
        
        while self.current_timestep <= total_timesteps:
            
            # start_all = time.time()
            if not self.imitation:
                eps = self.linear_eps(total_timesteps)
            
            # gather actions for robots from agents 
            # start_1 = time.time()
            actions = []
            for i,rob in enumerate(self.train_env.robots):
                if rob.deactivated:
                    actions.append(None)
                    continue
                
                if self.imitation:
                    # use imitation learning policy
                    action = self.il_agent.act(states[i])
                else:
                    if self.rl_agent.agent_type == "AC-IQN":
                        action = self.rl_agent.act_ac_iqn(states[i],eps,use_eval=False)
                    elif self.rl_agent.agent_type == "IQN":
                        action,_,_ = self.rl_agent.act_iqn(states[i],eps,use_eval=False)
                    elif self.rl_agent.agent_type == "DDPG":
                        action = self.rl_agent.act_ddpg(states[i],eps,use_eval=False)
                    elif self.rl_agent.agent_type == "DQN":
                        action = self.rl_agent.act_dqn(states[i],eps,use_eval=False)
                    elif self.rl_agent.agent_type == "SAC":
                        action = self.rl_agent.act_sac(states[i],eps,use_eval=False)
                    elif self.rl_agent.agent_type == "Rainbow":
                        action = self.rl_agent.act_rainbow(states[i],eps,use_eval=False)
                    else:
                        raise RuntimeError("Agent type not implemented!")

                actions.append(action)
            # end_1 = time.time()
            # elapsed_time_1 = end_1 - start_1
            # if self.current_timestep % 100 == 0:
            #     print("Elapsed time 1: {:.6f} seconds".format(elapsed_time_1))

            # start_2 = time.time()


            # execute actions in the training environment
            is_continuous_action = False
            if self.rl_agent.agent_type == "AC-IQN" or self.rl_agent.agent_type == "DDPG" or self.rl_agent.agent_type == "SAC":
                is_continuous_action = True
            next_states, rewards, dones, infos = self.train_env.step(actions,is_continuous_action)
            
            
            # end_2 = time.time()
            # elapsed_time_2 = end_2 - start_2
            # if self.current_timestep % 100 == 0:
            #     print("Elapsed time 2: {:.6f} seconds".format(elapsed_time_2))

            # save experience in replay memory
            for i,rob in enumerate(self.train_env.robots):
                if rob.deactivated:
                    continue

                ep_rewards[i] += self.rl_agent.GAMMA ** ep_length * rewards[i]
                if self.rl_agent.training:
                    if self.rl_agent.agent_type == "Rainbow":
                        self.rl_agent.memory.append(states[i], actions[i], rewards[i], dones[i])
                    else:
                        self.rl_agent.memory.add((states[i], actions[i], rewards[i], next_states[i], dones[i]))
                
                if rob.collision or rob.reach_goal:
                    rob.deactivated = True
                    ep_deactivated_t[i] = ep_length
                    
                    # Log robot deactivation
                    if self.logger is not None:
                        reason = "collision" if rob.collision else "goal_reached"
                        self.logger.log_robot_deactivation(
                            env_idx=0,
                            robot_idx=i,
                            reason=reason,
                            timestep=self.current_timestep,
                            episode_step=ep_length
                        )

            end_episode = (ep_length >= 1000) or self.train_env.check_all_deactivated()
            
            # Learn, update and evaluate models after learning_starts time step 
            if self.current_timestep >= self.learning_starts:
                # start_3 = time.time()

                if not self.rl_agent.training:
                    continue

                # Learn every UPDATE_EVERY time steps.
                if self.current_timestep % self.UPDATE_EVERY == 0:
                    # If enough samples are available in memory, get random subset and learn
                    num_elements = self.rl_agent.memory.transitions.num_elements() if self.rl_agent.agent_type == \
                                   "Rainbow" else self.rl_agent.memory.size()
                    
                    if num_elements > self.rl_agent.BATCH_SIZE:
                        self.rl_agent.train()

                # Update the target model every target_update_interval time steps
                if self.current_timestep % self.target_update_interval == 0:
                    self.rl_agent.soft_update()

                # end_3 = time.time()
                # elapsed_time_3 = end_3 - start_3
                # if self.current_timestep % 100 == 0:
                #     print("Elapsed time 3: {:.6f} seconds".format(elapsed_time_3))

                # Evaluate learning agents every eval_freq time steps
                if self.current_timestep == self.learning_starts or self.current_timestep % eval_freq == 0: 
                    self.evaluation()
                    self.save_evaluation(eval_log_path)

                    if not self.rl_agent.training:
                        continue
                    
                    # save the latest models
                    self.rl_agent.save_latest_model(eval_log_path)
                    
                    # Log checkpoint save
                    if self.logger is not None:
                        self.logger.log_checkpoint(
                            timestep=self.current_timestep,
                            checkpoint_path=eval_log_path
                        )

                # self.learning_timestep += 1

            if end_episode:
                # Log episode end
                if self.logger is not None:
                    success_info = {
                        i: (rob.reach_goal and not rob.collision)
                        for i, rob in enumerate(self.train_env.robots)
                    }
                    self.logger.log_episode_end(
                        env_idx=0,
                        episode_num=ep_num,
                        episode_length=ep_length,
                        rewards=ep_rewards,
                        success_info=success_info,
                        timestep=self.current_timestep
                    )
                
                ep_num += 1
                
                if verbose:
                    # print abstract info of the episode
                    if self.imitation:
                        print("======== IL Episode Info ========")
                    else:
                        print("======== RL Episode Info ========")

                    print("current ep_length: ",ep_length)
                    print("current ep_num: ",ep_num)
                    
                    if not self.imitation:
                        print("current exploration rate: ",eps)
                    
                    print("current timesteps: ",self.current_timestep)
                    print("total timesteps: ",total_timesteps)
                    print("======== Episode Info ========\n")
                    print("======== Robots Info ========")
                    for i,rob in enumerate(self.train_env.robots):
                        info = infos[i]["state"]
                        if info == "deactivated after collision" or info == "deactivated after reaching goal":
                            print(f"Robot {i} ep reward: {ep_rewards[i]:.2f}, {info} at step {ep_deactivated_t[i]}")
                        else:
                            print(f"Robot {i} ep reward: {ep_rewards[i]:.2f}, {info}")
                    print("======== Robots Info ========\n") 

                states,_,_ = self.train_env.reset()
                # cvar = 1 - np.random.uniform(0.0, 1.0)

                ep_rewards = np.zeros(len(self.train_env.robots))
                ep_deactivated_t = [-1]*len(self.train_env.robots)
                ep_length = 0
                
                # Log new episode start
                if self.logger is not None:
                    self.logger.log_episode_start(
                        env_idx=0,
                        episode_num=ep_num,
                        num_robots=len(self.train_env.robots),
                        timestep=self.current_timestep
                    )
            else:
                states = next_states
                ep_length += 1

            # end_all = time.time()
            # elapsed_time_all = end_all - start_all
            # if self.current_timestep % 100 == 0:
            #     print("one step elapsed time: {:.6f} seconds".format(elapsed_time_all))
            
            self.current_timestep += 1

    def linear_eps(self,total_timesteps):
        
        progress = self.current_timestep / total_timesteps
        if progress < self.exploration_fraction:
            r = progress / self.exploration_fraction
            return self.initial_eps + r * (self.final_eps - self.initial_eps)
        else:
            return self.final_eps

    def _learn_vec_env(self,
              total_timesteps,
              eval_freq,
              eval_log_path,
              verbose=True):
        """Learning loop for vectorized environments (multiple parallel environments)"""
        
        # Reset all environments
        states_list = self.train_env.reset()  # List of state arrays, one per env
        
        # Track episode state for each environment
        ep_rewards_list = []
        ep_deactivated_t_list = []
        ep_length_list = []
        ep_num_list = []
        
        # Initialize episode tracking for each environment
        for env_idx, states in enumerate(states_list):
            num_robots = len(states)
            ep_rewards_list.append(np.zeros(num_robots))
            ep_deactivated_t_list.append([-1] * num_robots)
            ep_length_list.append(0)
            ep_num_list.append(0)
            
            # Log episode start for each environment
            if self.logger is not None:
                self.logger.log_episode_start(
                    env_idx=env_idx,
                    episode_num=0,
                    num_robots=num_robots,
                    timestep=self.current_timestep
                )
        
        while self.current_timestep <= total_timesteps:
            
            if not self.imitation:
                eps = self.linear_eps(total_timesteps)
            
            # Gather actions for all robots across all environments
            # Collect all states for batch inference
            all_states = []
            state_indices = []  # Track which env and robot each state belongs to
            
            for env_idx, states in enumerate(states_list):
                for robot_idx, state in enumerate(states):
                    all_states.append(state)
                    state_indices.append((env_idx, robot_idx))
            
            # Batch inference for all states at once
            if len(all_states) > 0:
                if self.imitation:
                    all_actions = [self.il_agent.act(state) for state in all_states]
                else:
                    # TODO: Implement true batch inference in agent methods
                    # For now, still iterate but at least it's cleaner
                    all_actions = []
                    for state in all_states:
                        if self.rl_agent.agent_type == "AC-IQN":
                            action = self.rl_agent.act_ac_iqn(state, eps, use_eval=False)
                        elif self.rl_agent.agent_type == "IQN":
                            action, _, _ = self.rl_agent.act_iqn(state, eps, use_eval=False)
                        elif self.rl_agent.agent_type == "DDPG":
                            action = self.rl_agent.act_ddpg(state, eps, use_eval=False)
                        elif self.rl_agent.agent_type == "DQN":
                            action = self.rl_agent.act_dqn(state, eps, use_eval=False)
                        elif self.rl_agent.agent_type == "SAC":
                            action = self.rl_agent.act_sac(state, eps, use_eval=False)
                        elif self.rl_agent.agent_type == "Rainbow":
                            action = self.rl_agent.act_rainbow(state, eps, use_eval=False)
                        else:
                            raise RuntimeError("Agent type not implemented!")
                        all_actions.append(action)
            
            # Reorganize actions back into per-environment lists
            actions_list = [[] for _ in range(self.n_envs)]
            for action, (env_idx, robot_idx) in zip(all_actions, state_indices):
                actions_list[env_idx].append(action)
            
            # Execute actions in all training environments
            is_continuous_action = False
            if self.rl_agent.agent_type in ["AC-IQN", "DDPG", "SAC"]:
                is_continuous_action = True
            
            next_states_list, rewards_list, dones_list, infos_list = self.train_env.step(
                actions_list, is_continuous_action
            )
            
            # Get robot info from all environments
            robots_info_list = self.train_env.get_robots_info()
            
            # Process experiences from all environments
            for env_idx in range(self.n_envs):
                states = states_list[env_idx]
                actions = actions_list[env_idx]
                rewards = rewards_list[env_idx]
                next_states = next_states_list[env_idx]
                dones = dones_list[env_idx]
                infos = infos_list[env_idx]
                robots_info = robots_info_list[env_idx]
                
                ep_length = ep_length_list[env_idx]
                ep_rewards = ep_rewards_list[env_idx]
                ep_deactivated_t = ep_deactivated_t_list[env_idx]
                
                # Save experience in replay memory
                for i in range(len(states)):
                    if robots_info[i]['deactivated']:
                        continue
                    
                    ep_rewards[i] += self.rl_agent.GAMMA ** ep_length * rewards[i]
                    
                    if self.rl_agent.training:
                        if self.rl_agent.agent_type == "Rainbow":
                            self.rl_agent.memory.append(states[i], actions[i], rewards[i], dones[i])
                        else:
                            self.rl_agent.memory.add((states[i], actions[i], rewards[i], next_states[i], dones[i]))
                    
                    if robots_info[i]['collision'] or robots_info[i]['reach_goal']:
                        ep_deactivated_t[i] = ep_length
                        
                        # Log robot deactivation
                        if self.logger is not None:
                            reason = "collision" if robots_info[i]['collision'] else "goal_reached"
                            self.logger.log_robot_deactivation(
                                env_idx=env_idx,
                                robot_idx=i,
                                reason=reason,
                                timestep=self.current_timestep,
                                episode_step=ep_length
                            )
                
                # Check if episode ended for this environment
                # Get environment info
                env_info_list = self.train_env.get_env_info()
                env_info = env_info_list[env_idx]
                
                end_episode = (ep_length >= 1000) or env_info['check_all_deactivated']
                
                if end_episode:
                    # Log episode end
                    if self.logger is not None:
                        success_info = {
                            i: (robots_info[i]['reach_goal'] and not robots_info[i]['collision'])
                            for i in range(len(robots_info))
                        }
                        self.logger.log_episode_end(
                            env_idx=env_idx,
                            episode_num=ep_num_list[env_idx],
                            episode_length=ep_length,
                            rewards=ep_rewards,
                            success_info=success_info,
                            timestep=self.current_timestep
                        )
                    
                    ep_num_list[env_idx] += 1
                    
                    if verbose and env_idx == 0:  # Only print info for first env to avoid spam
                        if self.imitation:
                            print("======== IL Episode Info ========")
                        else:
                            print("======== RL Episode Info ========")
                        
                        print(f"Env {env_idx} - current ep_length: ", ep_length)
                        print(f"Env {env_idx} - current ep_num: ", ep_num_list[env_idx])
                        
                        if not self.imitation:
                            print("current exploration rate: ", eps)
                        
                        print("current timesteps: ", self.current_timestep)
                        print("total timesteps: ", total_timesteps)
                        print("======== Episode Info ========\n")
                        print("======== Robots Info ========")
                        for i in range(len(infos)):
                            info = infos[i]["state"]
                            if "deactivated after collision" in info or "deactivated after reaching goal" in info:
                                print(f"Robot {i} ep reward: {ep_rewards[i]:.2f}, {info} at step {ep_deactivated_t[i]}")
                            else:
                                print(f"Robot {i} ep reward: {ep_rewards[i]:.2f}, {info}")
                        print("======== Robots Info ========\n")
                    
                    # Reset episode tracking for this environment
                    ep_rewards_list[env_idx] = np.zeros(len(states))
                    ep_deactivated_t_list[env_idx] = [-1] * len(states)
                    ep_length_list[env_idx] = 0
                    
                    # Log new episode start
                    if self.logger is not None:
                        self.logger.log_episode_start(
                            env_idx=env_idx,
                            episode_num=ep_num_list[env_idx],
                            num_robots=len(states),
                            timestep=self.current_timestep
                        )
                else:
                    ep_length_list[env_idx] += 1
            
            # Update states for next iteration
            states_list = next_states_list
            
            # Learn, update and evaluate models after learning_starts time step
            if self.current_timestep >= self.learning_starts:
                
                if not self.rl_agent.training:
                    continue
                
                # Learn every UPDATE_EVERY time steps
                if self.current_timestep % self.UPDATE_EVERY == 0:
                    num_elements = self.rl_agent.memory.transitions.num_elements() if self.rl_agent.agent_type == \
                                   "Rainbow" else self.rl_agent.memory.size()
                    
                    if num_elements > self.rl_agent.BATCH_SIZE:
                        self.rl_agent.train()
                
                # Update the target model every target_update_interval time steps
                if self.current_timestep % self.target_update_interval == 0:
                    self.rl_agent.soft_update()
                
                # Evaluate learning agents every eval_freq time steps
                if self.current_timestep == self.learning_starts or self.current_timestep % eval_freq == 0:
                    self.evaluation()
                    self.save_evaluation(eval_log_path)
                    
                    if not self.rl_agent.training:
                        continue
                    
                    # save the latest models
                    self.rl_agent.save_latest_model(eval_log_path)
                    
                    # Log checkpoint save
                    if self.logger is not None:
                        self.logger.log_checkpoint(
                            timestep=self.current_timestep,
                            checkpoint_path=eval_log_path
                        )
            
            self.current_timestep += 1

    def evaluation(self):
        """Evaluate performance of the RL agent
        Params
        ======
            eval_env (gym compatible env): evaluation environment
            eval_config: eval envs config file
        """
        observations_data = []
        actions_data = []
        trajectories_data = []
        rewards_data = []
        successes_data = []
        times_data = []
        energies_data = []
        relations_data = []
        # obs_data = []
        # objs_data = []
        
        for idx, config in enumerate(self.eval_config):
            print(f"Evaluating episode {idx}")
            state,_,_ = self.eval_env.reset_with_eval_config(config)
            # obs = [[copy.deepcopy(rob.perception.observed_obs)] for rob in self.eval_env.robots]
            # objs = [[copy.deepcopy(rob.perception.observed_objs)] for rob in self.eval_env.robots]
            
            rob_num = len(self.eval_env.robots)

            relations = [[] for _ in range(rob_num)]
            rewards = [0.0]*rob_num
            times = [0.0]*rob_num
            energies = [0.0]*rob_num
            end_episode = False
            length = 0
            
            while not end_episode:
                # gather actions for robots from RL agents 
                action = []
                for i,rob in enumerate(self.eval_env.robots):
                    if rob.deactivated:
                        action.append(None)
                        continue
                    
                    if self.rl_agent.agent_type == "AC-IQN":
                        a = self.rl_agent.act_ac_iqn(state[i])
                    elif self.rl_agent.agent_type == "IQN":
                        a,_,_ = self.rl_agent.act_iqn(state[i])
                    elif self.rl_agent.agent_type == "DDPG":
                        a = self.rl_agent.act_ddpg(state[i])
                    elif self.rl_agent.agent_type == "DQN":
                        a = self.rl_agent.act_dqn(state[i])
                    elif self.rl_agent.agent_type == "SAC":
                        a = self.rl_agent.act_sac(state[i])
                    elif self.rl_agent.agent_type == "Rainbow":
                        a = self.rl_agent.act_rainbow(state[i])
                    else:
                        raise RuntimeError("Agent type not implemented!")                 

                    action.append(a)

                # execute actions in the training environment
                is_continuous_action = False
                if self.rl_agent.agent_type == "AC-IQN" or self.rl_agent.agent_type == "DDPG" or self.rl_agent.agent_type == "SAC":
                    is_continuous_action = True
                
                state, reward, done, info = self.eval_env.step(action,is_continuous_action)
                
                for i,rob in enumerate(self.eval_env.robots):
                    if rob.deactivated:
                        continue
                    
                    rewards[i] += self.rl_agent.GAMMA ** length * reward[i]
                    times[i] += rob.dt * rob.N
                    energies[i] += rob.compute_step_energy_cost()
                    # obs[i].append(copy.deepcopy(rob.perception.observed_obs))
                    # objs[i].append(copy.deepcopy(rob.perception.observed_objs))

                    if rob.collision or rob.reach_goal:
                        rob.deactivated = True

                end_episode = (length >= 1000) or self.eval_env.check_any_collision() or self.eval_env.check_all_deactivated()
                length += 1

            observations = []
            actions = []
            trajectories = []
            for rob in self.eval_env.robots:
                observations.append(copy.deepcopy(rob.observation_history))
                actions.append(copy.deepcopy(rob.action_history))
                trajectories.append(copy.deepcopy(rob.trajectory))

            success = True if self.eval_env.check_all_reach_goal() else False

            observations_data.append(observations)
            actions_data.append(actions)
            trajectories_data.append(trajectories)
            rewards_data.append(np.mean(rewards))
            successes_data.append(success)
            times_data.append(np.mean(times))
            energies_data.append(np.mean(energies))
            relations_data.append(relations)
            # obs_data.append(obs)
            # objs_data.append(objs)
        
        avg_r = np.mean(rewards_data)
        success_rate = np.sum(successes_data)/len(successes_data)
        idx = np.where(np.array(successes_data) == 1)[0]
        avg_t = None if np.shape(idx)[0] == 0 else np.mean(np.array(times_data)[idx])
        avg_e = None if np.shape(idx)[0] == 0 else np.mean(np.array(energies_data)[idx])

        print(f"++++++++ Evaluation Info ++++++++")
        print(f"Avg cumulative reward: {avg_r:.2f}")
        print(f"Success rate: {success_rate:.2f}")
        if avg_t is not None:
            print(f"Avg time: {avg_t:.2f}")
            print(f"Avg energy: {avg_e:.2f}")
        print(f"++++++++ Evaluation Info ++++++++\n")
        
        # Log evaluation results
        if self.logger is not None:
            self.logger.log_evaluation(
                timestep=self.current_timestep,
                avg_reward=avg_r,
                success_rate=success_rate,
                avg_time=avg_t,
                avg_energy=avg_e
            )

        self.eval_timesteps.append(self.current_timestep)
        self.eval_observations.append(observations_data)
        self.eval_actions.append(actions_data)
        self.eval_trajectories.append(trajectories_data)
        self.eval_rewards.append(rewards_data)
        self.eval_successes.append(successes_data)
        self.eval_times.append(times_data)
        self.eval_energies.append(energies_data)
        self.eval_relations.append(relations_data)
        # self.eval_obs.append(obs_data)
        # self.eval_objs.append(objs_data)

    def save_evaluation(self,eval_log_path):
        filename = "evaluations.npz"
        
        np.savez(
            os.path.join(eval_log_path,filename),
            timesteps=np.array(self.eval_timesteps,dtype=object),
            observations=np.array(self.eval_observations,dtype=object),
            actions=np.array(self.eval_actions,dtype=object),
            trajectories=np.array(self.eval_trajectories,dtype=object),
            rewards=np.array(self.eval_rewards,dtype=object),
            successes=np.array(self.eval_successes,dtype=object),
            times=np.array(self.eval_times,dtype=object),
            energies=np.array(self.eval_energies,dtype=object),
            relations=np.array(self.eval_relations,dtype=object),
            # obs=self.eval_obs,
            # objs=self.eval_objs
        )