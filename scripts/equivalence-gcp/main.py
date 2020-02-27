import numpy as np
import gym
import torch.nn as nn
import torch
from torch.optim import Adam
import argparse
import os
import errno
import pickle
import time
from random import shuffle
from utils import set_global_seed, distance, rotate_point
from utils import find_norm, prepare_dir, normalize, preprocess_goal, save_checkpoint
from helper_functions import load_checkpoint
from network import Encoder, Policy
from tensorboardX import SummaryWriter
import pybullet_envs

parser = argparse.ArgumentParser(description='Equivalence')
parser.add_argument('-e', '--env-name', default='Ant-v2',
                    help='Environment name')
parser.add_argument('-i', '--input-file', default='',
                    help='Path to file containing trajectories.')
parser.add_argument('--resume', default='',
                    help='Previous checkpoint path, from which training is to '
                    'be resumed.')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('-s', '--seed', type=int, default=0,
                    help='Seed for initializing the network.')
parser.add_argument('--no-gpu', action='store_true')
parser.add_argument('--dir-name', default='', help='')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--n-test-steps', type=int, default=10,
                    help='Number of test trajectories')
parser.add_argument('--log-perf-file', default='',
                    help='File in which results of current run are to be '
                    ' stored')
parser.add_argument('--min-distance', type=float, default=0.5,
                    help='Min. distance to target')
parser.add_argument('--max-distance', type=float, default=1.0,
                    help='Max. distance to target')
parser.add_argument('--threshold', type=float, default=0.1,
                    help='Threshold distance for navigation to be considered '
                    'successful')
parser.add_argument('--y-range', type=float, default=0.01,
                    help='Max. distance along +ve and -ve y-axis that the '
                    'target can lie')
parser.add_argument('--n-training-samples', type=int, default=1000000,
                    help='Number of samples used to train the models')
parser.add_argument('--start-index', type=int, default=0,
                    help='Starting index of transitions in pickle file')
parser.add_argument('--exp-name', default='exp-3.2.16',
                    help='Alias for the experiment')
parser.add_argument('--batch-size', default=512, type=int,
                    help='Batch size')
parser.add_argument('--learning-rate', default=0.001, type=float,
                    help='Learning rate')
parser.add_argument('--n-epochs', type=int, default=1000,
                    help='Number of epochs')
parser.add_argument('--encoder-dim', type=int, default=10,
                    help='Dimension k of the embedding generated by the encoder')
parser.add_argument('--lam', type=float, default=0.25,
                    help='Lambda: proportion of encoder loss to policy loss')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    args = parser.parse_args()
    env_name = args.env_name
    input_file = args.input_file
    checkpoint_file = args.resume
    test_only = args.test_only
    seed = args.seed
    dir_name = args.dir_name
    visualize = args.visualize
    n_test_steps = args.n_test_steps
    log_perf_file = args.log_perf_file
    min_distance = args.min_distance
    max_distance = args.max_distance
    threshold = args.threshold
    y_range = args.y_range
    n_training_samples = args.n_training_samples
    start_index = args.start_index
    exp_name = args.exp_name
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    n_epochs = args.n_epochs
    encoder_dim = args.encoder_dim
    lam = args.lam

    # Specific to Humanoid - Pybullet
    if visualize and env_name == 'HumanoidBulletEnv-v0':
        spec = gym.envs.registry.env_specs[env_name]
        class_ = gym.envs.registration.load(spec._entry_point)
        env = class_(**{**spec._kwargs}, **{'render': True})
    else:
        env = gym.make(env_name)

    set_global_seed(seed)
    env.seed(seed)

    input_shape = env.observation_space.shape[0] + 3
    output_shape = env.action_space.shape[0]
    encoder = Encoder(input_shape, encoder_dim).to(device)
    policy = Policy(encoder_dim, output_shape).to(device)
    optimizer = Adam(list(encoder.parameters()) +
                     list(policy.parameters()),
                     lr=learning_rate)
    encoder_criterion = nn.MSELoss()
    policy_criterion = nn.MSELoss()
    epochs = 0

    if checkpoint_file:
        (epochs,
         encoder,
         policy,
         optimizer) = load_checkpoint(checkpoint_file,
                                      encoder,
                                      policy,
                                      optimizer)

    if not checkpoint_file and test_only:
        print('ERROR: You have not entered a checkpoint file.')
        return

    if not test_only:
        if not os.path.isfile(input_file):
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT),
                                    input_file)

        training_file = open(input_file, 'rb')
        old_states, aug_old_states = [], []
        norms = []
        goals, aug_goals = [], []
        actions = []
        n_samples = -1

        while n_samples - start_index < n_training_samples:
            try:
                (old_s,
                 old_g,
                 new_s,
                 new_g,
                 action) = pickle.load(training_file)
                (aug_old_s,
                 aug_old_g,
                 aug_new_s,
                 aug_new_g,
                 _) = pickle.load(training_file)
                n_samples += 2

                if n_samples < start_index:
                    continue

                old_states.append(np.squeeze(np.array(old_s)))
                aug_old_states.append(np.squeeze(np.array(aug_old_s)))
                norms.append(find_norm(np.squeeze(np.array(new_g) -
                                                  np.array(old_g))))
                norms.append(find_norm(np.squeeze(np.array(aug_new_g) -
                                                  np.array(aug_old_g))))
                goals.append(preprocess_goal(np.squeeze(
                                                    np.array(new_g) -
                                                    np.array(old_g))))
                aug_goals.append(preprocess_goal(np.squeeze(
                                                        np.array(aug_new_g) -
                                                        np.array(aug_old_g))))
                actions.append(np.squeeze(np.array(action)))

            except (EOFError, ValueError):
                break

        old_states = np.array(old_states)
        aug_old_states = np.array(aug_old_states)
        norms = np.array(norms)
        goals = np.array(goals)
        aug_goals = np.array(aug_goals)
        actions = np.array(actions)

        normalization_factors = {'state':
                                 [np.concatenate((old_states, aug_old_states),
                                                 axis=0).mean(axis=0),
                                  np.concatenate((old_states, aug_old_states),
                                                 axis=0).std(axis=0)],
                                 'distance_per_step':
                                 [norms.mean(axis=0),
                                  norms.std(axis=0)],
                                 'action':
                                 [actions.mean(axis=0),
                                  actions.std(axis=0)]}
        n_file = open(env_name + '_normalization_factors.pkl', 'wb')
        pickle.dump(normalization_factors, n_file)
        n_file.close()

        old_states = normalize(old_states,
                               env_name + '_normalization_factors.pkl',
                               'state')

        aug_old_states = normalize(aug_old_states,
                                   env_name + '_normalization_factors.pkl',
                                   'state')

        # Summary writer for tensorboardX
        writer = {}
        writer['writer'] = SummaryWriter()

        # Split data into training and validation
        indices = np.arange(old_states.shape[0])
        shuffle(indices)
        val_indices = indices[:int(indices.shape[0]/5)]
        training_indices = indices[int(indices.shape[0]/5):]

        # Creating checkpoint (sub)directories if they do not exist.
        checkpoint_dir = os.path.join(env_name, 'equivalence_checkpoints')
        if dir_name:
            checkpoint_dir = os.path.join(checkpoint_dir, dir_name)
        prepare_dir(checkpoint_dir)

        for e in range(epochs, n_epochs):
            # Train network
            ep_loss = []
            c1_loss, c2_loss = [], []
            for i in range(int(np.ceil((len(training_indices) / batch_size)))):
                aug_inp = np.concatenate(
                            (aug_old_states[training_indices[batch_size * i:
                                            batch_size * (i + 1)]],
                             aug_goals[training_indices[
                                                    batch_size * i:
                                                    batch_size * (i + 1)]]),
                            axis=1)
                aug_inp = torch.from_numpy(aug_inp).float().to(device)
                inp = np.concatenate(
                                     (old_states[training_indices[batch_size * i:
                                                 batch_size * (i + 1)]],
                                      goals[training_indices[batch_size * i:
                                            batch_size * (i + 1)]]),
                              axis=1)
                inp = torch.from_numpy(inp).float().to(device)

                target = actions[training_indices[batch_size * i:
                                 batch_size * (i + 1)]]
                target = torch.from_numpy(target).float().to(device)

                embedding = encoder(inp)
                aug_embedding = encoder(aug_inp)
                mean_encoder = (embedding + aug_embedding) / 2
                c1 = encoder_criterion(embedding, aug_embedding)

                pred_action = policy(embedding)
                aug_pred_action = policy(aug_embedding)
                mean_pred_action = policy(mean_encoder)
                c2 = policy_criterion(mean_pred_action, target)

                loss = (lam * c1 + (1 - lam) * c2)
                optimizer.zero_grad()
                ep_loss.append(loss.item())
                c1_loss.append(c1.item())
                c2_loss.append(c2.item())
                loss.backward()
                optimizer.step()

            # Validation
            val_loss = []
            (val_c1_loss, val_c2_loss) = [], []
            for i in range(int(len(val_indices) / batch_size) + 1):
                aug_inp = np.concatenate(
                            (aug_old_states[val_indices[batch_size * i:
                                            batch_size * (i + 1)]],
                             aug_goals[val_indices[
                                                    batch_size * i:
                                                    batch_size * (i + 1)]]),
                            axis=1)
                aug_inp = torch.from_numpy(aug_inp).float().to(device)
                inp = np.concatenate(
                              (old_states[val_indices[batch_size * i:
                                          batch_size * (i + 1)]],
                               goals[val_indices[
                                                  batch_size * i:
                                                  batch_size * (i + 1)]]),
                              axis=1)
                inp = torch.from_numpy(inp).float().to(device)

                target = actions[val_indices[batch_size * i:
                                 batch_size * (i + 1)]]
                target = torch.from_numpy(target).float().to(device)

                embedding = encoder(inp)
                aug_embedding = encoder(aug_inp)
                mean_encoder = (embedding + aug_embedding) / 2
                c1 = encoder_criterion(embedding, aug_embedding)

                pred_action = policy(embedding)
                aug_pred_action = policy(aug_embedding)
                mean_pred_action = policy(mean_encoder)
                c2 = policy_criterion(mean_pred_action, target)

                loss = (lam * c1 + (1 - lam) * c2)
                val_loss.append(loss.item())
                val_c1_loss.append(c1.item())
                val_c2_loss.append(c2.item())

            writer['iter'] = e + 1
            writer['writer'].add_scalar('data/val_loss',
                                        np.array(val_loss).mean(),
                                        e + 1)
            writer['writer'].add_scalar('data/val_c1_loss',
                                        np.array(val_c1_loss).mean(),
                                        e + 1)
            writer['writer'].add_scalar('data/val_c2_loss',
                                        np.array(val_c2_loss).mean(),
                                        e + 1)

            writer['writer'].add_scalar('data/training_loss',
                                        np.array(ep_loss).mean(),
                                        e + 1)
            writer['writer'].add_scalar('data/training_c1_loss',
                                        np.array(c1_loss).mean(),
                                        e + 1)
            writer['writer'].add_scalar('data/training_c2_loss',
                                        np.array(c2_loss).mean(),
                                        e + 1)

            save_checkpoint({'epochs': (e + 1),
                             'encoder_state_dict': encoder.state_dict(),
                             'policy_state_dict': policy.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            filename=os.path.join(checkpoint_dir,
                                                  str(e + 1) + '.pth.tar'))

            print('Epoch:', e + 1)
            print('Training loss:', np.array(ep_loss).mean())
            print('Val loss:', np.array(val_loss).mean())
            print('')

    # Now we use the trained net to see how the agent reaches a different
    # waypoint from the current one.

    success = 0
    failure = 0

    closest_distances = []
    time_to_closest_distances = []

    for i in range(n_test_steps):
        state = env.reset()
        if env_name == 'Ant-v2':
            obs = env.unwrapped.get_body_com('torso')
            target_obs = [obs[0] + np.random.uniform(min_distance, max_distance),
                          obs[1] + np.random.uniform(-y_range, y_range),
                          obs[2]]
            target_obs = rotate_point(target_obs, env.unwrapped.angle)
            env.unwrapped.sim.model.body_pos[-1] = target_obs
        elif env_name == 'MinitaurBulletEnv-v0':
            obs = env.unwrapped.get_minitaur_position()
            target_obs = [obs[0] + np.random.uniform(min_distance, max_distance),
                          obs[1] + np.random.uniform(-y_range, y_range),
                          obs[2]]
            target_obs = rotate_point(target_obs, env.unwrapped.get_minitaur_rotation_angle())
            env.unwrapped.set_target_position(target_obs)
        elif env_name == 'HumanoidBulletEnv-v0':
            obs = env.unwrapped.robot.get_robot_position()
            target_obs = [obs[0] + np.random.uniform(min_distance, max_distance),
                          obs[1] + np.random.uniform(-y_range, y_range),
                          obs[2]]
            target_obs = rotate_point(target_obs, env.unwrapped.robot.yaw)
            env.unwrapped.robot.set_target_position(target_obs[0], target_obs[1])
        steps = 0
        done = False
        closest_d = distance(obs, target_obs)
        closest_t = 0

        while distance(obs, target_obs) > threshold and not done:
            goal = preprocess_goal(target_obs - obs)
            state = normalize(state,
                              n_file=env_name+'_normalization_factors.pkl',
                              key='state')
            inp = np.concatenate((np.squeeze(state),
                                  np.squeeze(goal)))
            inp = torch.from_numpy(inp).float().to(device)
            embedding = encoder(inp)
            action = policy(embedding).to('cpu').detach().numpy()
            state, _, done, _ = env.step(action)
            steps += 1
            if env_name == 'MinitaurBulletEnv-v0':
                obs = env.unwrapped.get_minitaur_position()
            elif env_name == 'HumanoidBulletEnv-v0':
                obs = env.unwrapped.robot.get_robot_position()
            if distance(obs, target_obs) < closest_d:
                closest_d = distance(obs, target_obs)
                closest_t = steps
            if visualize:
                env.render()

        if distance(obs, target_obs) <= threshold:
            success += 1
        elif done:
            failure += 1

        if visualize:
            time.sleep(2)

        closest_distances.append(closest_d)
        time_to_closest_distances.append(closest_t)

    print('Successes: %d, Failures: %d, '
          'Closest distance: %f, Time to closest distance: %d'
          % (success, failure, np.mean(closest_distances),
             np.mean(time_to_closest_distances)))

    if log_perf_file:
        f = open(log_perf_file, 'a+')
        f.write(exp_name + ':Seed-' + str(seed) + ',Success-' +
                str(success) + ',Failure-' + str(failure) +
                ',Closest_distance-' + str(closest_distances) +
                ',Time_to_closest_distance-' + str(time_to_closest_distances)
                + '\n')
        f.close()


if __name__ == '__main__':
    main()