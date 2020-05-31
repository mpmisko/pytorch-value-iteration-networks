import sys

import numpy as np
from dataset import *

sys.path.append('.')
from generators.obstacle_gen import *
sys.path.remove('.')


def extract_action(traj):
    # Given a trajectory, outputs a 1D vector of
    #  actions corresponding to the trajectory.
    n_actions = 8
    action_vecs = np.asarray([[-1., 0.], [1., 0.], [0., 1.], [0., -1.],
                              [-1., 1.], [-1., -1.], [1., 1.], [1., -1.]])
    action_vecs[4:] = 1 / np.sqrt(2) * action_vecs[4:]
    action_vecs = action_vecs.T
    state_diff = np.diff(traj, axis=0)
    norm_state_diff = state_diff * np.tile(
        1 / np.sqrt(np.sum(np.square(state_diff), axis=1)), (2, 1)).T
    prj_state_diff = np.dot(norm_state_diff, action_vecs)
    actions_one_hot = np.abs(prj_state_diff - 1) < 0.00001
    actions = np.dot(actions_one_hot, np.arange(n_actions).T)
    return actions


def make_data(dom_size, n_domains, max_obs, max_obs_size, n_traj,
              state_batch_size):

    images = []
    obs_vectors = []

    dom = 0.0
    while dom < n_domains:
        goal = [np.random.randint(dom_size[0]), np.random.randint(dom_size[1])]
        # Generate obstacle map
        obs = obstacles([dom_size[0], dom_size[1]], goal, max_obs_size)
        # Add obstacles to map
        n_obs = obs.add_n_rand_obs(max_obs)
        # Add border to map
        border_res = obs.add_border()

        # Ensure we have valid map
        if n_obs == 0 or not border_res:
            continue
        # Get final map

        im, obs_vec = obs.get_final()
        obs.show()
        obs_vectors.append(np.array(obs_vec))
        im = 1 - im
        images.append(im)

        dom += 1
        sys.stdout.write("\r" + str(int((dom / n_domains) * 100)) + "%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    # Concat all outputs
    images = np.array(images)
    obs_vectors = np.array(obs_vectors)
    return images, obs_vectors


def main(dom_size=[28, 28],
         n_domains=100000,
         max_obs=2,
         max_obs_size=8,
         n_traj=7,
         state_batch_size=1):
    # Get path to save dataset
    save_path = "dataset/gridworld_{0}x{1}".format(dom_size[0], dom_size[1])
    # Get training data
    print("Now making training data...")
    images_tr, obs_vectors_tr = make_data(dom_size, n_domains, max_obs, max_obs_size, n_traj, state_batch_size)
    # Get testing data
    print("\nNow making  testing data...")
    images_ts, obs_vectors_ts = make_data(
        dom_size, n_domains / 10, max_obs, max_obs_size, n_traj,
        state_batch_size)

    # Save dataset
    np.savez_compressed(save_path, images_tr, obs_vectors_tr, images_ts, obs_vectors_ts)


if __name__ == '__main__':
    main()
