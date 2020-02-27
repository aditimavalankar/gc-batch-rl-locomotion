import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Distance metric')
parser.add_argument('-e', '--env-name', default='Ant-v2',
                    help='Environment name')
parser.add_argument('-s', '--seed', default=0, help='Random seed')
parser.add_argument('--ckpt-dir', default='../checkpoints/',
                    help='Path to directory containing checkpoints')

checkpoints = {
                'Random samples': 'goal-conditioned-batch-rl_random.pth.tar',
                'On-policy samples': 'goal-conditioned-batch-rl_on-policy.pth.tar',
                'Augmented samples': 'goal-conditioned-batch-rl_augmented.pth.tar',
                'Equivalence': 'goal-conditioned-batch-rl_augmented.pth.tar'
              }

test_parameters = {
                   'Ant-v2': {'min_distance': 2.0,
                              'max_distance': 4.0,
                              'y_range': 2.0,
                              'threshold': 0.0
                              },
                   'MinitaurBulletEnv-v0': {'min_distance': 1.0,
                                            'max_distance': 2.0,
                                            'y_range': 1.0,
                                            'threshold': 0.0
                                            },
                   'HumanoidBulletEnv-v0': {'min_distance': 2.0,
                                            'max_distance': 4.0,
                                            'y_range': 2.0,
                                            'threshold': 0.0
                                            }
                  }


def calculate_results(env_name, seeds, ckpt_dir):
    """
    This function tests all the goal-conditioned batch RL methods on the
    goal-directed locomotion task and records the results in a log file.
    """
    base_dir = os.getcwd()
    log_file = os.path.join(base_dir, 'closest_distance_metric_%s.txt' % (env_name))
    for c in checkpoints:
        print(c)
        os.chdir(ckpt_dir)
        for seed in seeds:
            command = ('python3 main.py --test-only '
                       '--env-name %s '
                       '--resume %s '
                       '--seed %d '
                       '--n-test-steps 100 '
                       '--log-perf-file %s '
                       '--min-distance %f '
                       '--max-distance %f '
                       '--y-range %f '
                       '--threshold %f '
                       '--exp-name %s '
                       % (env_name,
                          os.path.join(env_name,
                                       checkpoints[c]),
                          seed,
                          log_file,
                          test_parameters[env_name]['min_distance'],
                          test_parameters[env_name]['max_distance'],
                          test_parameters[env_name]['y_range'],
                          test_parameters[env_name]['threshold'],
                          c))
            os.system(command)
        os.chdir(base_dir)


def analyze_results(env_name):
    """
    This function displays the mean and standard deviation of the
    goal-conditioned batch RL methods from the log file created by the
    calculate_results() function.
    """
    fname = 'closest_distance_metric_' + env_name + '.txt'
    f = open(fname, 'r')
    lines = f.read().split('\n')
    results = {}

    for c in checkpoints:
        results[c] = {'success': [],
                      'failure': [],
                      'closest_distance': [],
                      'closest_time': []}

    for l in lines[:-1]:
        key, rem = l.split(':')
        s1, s2, s3 = rem.split('[')
        _, success, death, _ = s1.split(',')
        closest_distance = list(map(float, s2.split(']')[0].split(',')))
        closest_time = list(map(float, s3.split(']')[0].split(',')))
        success = int(success.split('-')[1])
        death = int(death.split('-')[1])
        results[key]['success'].append(success)
        results[key]['failure'].append(death)
        results[key]['closest_distance'] += closest_distance
        results[key]['closest_time'] += closest_time

    f.close()

    print('Exp', '\t\t',
          'Success', '\t',
          'Failure', '\t',
          'Closest distance', '\n')

    for c in checkpoints:
        print(c, '\t',
              np.array(results[c]['success']).mean(), '+-',
              np.array(results[c]['success']).std(), '\t',
              np.array(results[c]['failure']).mean(), '+-',
              np.array(results[c]['failure']).std(), '\t',
              np.array(results[c]['closest_distance']).mean(), '+-',
              np.array(results[c]['closest_distance']).std(), '\n')


if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)
    seeds = np.random.randint(1000000, size=10)
    print('Random seeds:', seeds)
    calculate_results(args.env_name, seeds, args.ckpt_dir)
    analyze_results(args.env_name)
