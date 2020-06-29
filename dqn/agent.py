from __future__ import print_function
import random
import numpy as np
from tqdm import tqdm
from functools import reduce
from .history import History
from .replay_memory import ReplayMemory
from .utils import get_time, save_pkl, load_pkl
import os
import tensorflow as tf


class BaseModel(object):
    """Abstract object representing an Reader model."""

    def __init__(self, config, sess):
        self.saver = None
        self.sess = sess
        self.config = config
        self.log_dir = os.path.join(os.path.abspath(os.curdir), config.model_name)
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')

    def save_model(self, name=None, step=None):
        print("[*****] Saving checkpoints...")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if name is None:
            name = os.path.join(self.checkpoint_dir, "{}_{}".format(self.config.model_name, step))
        self.saver.save(self.sess, name, global_step=step)

    def load_model(self):
        print("[*****] Loading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            fname = os.path.join(self.checkpoint_dir, "/{}.best.model".format(self.config.model_name))
            self.saver.restore(self.sess, fname)
            print("[*****] Loading SUCCESS: %s" % fname)
            return True
        else:
            print("[*****] Loading FAILED: %s" % self.checkpoint_dir)
            return False


class Agent(BaseModel):
    def __init__(self, config, environment, sess):
        super(Agent, self).__init__(config, sess)
        self.weight_dir = os.path.join(self.log_dir, 'weights')
        self.env = environment
        self.history = History(self.config)
        self.memory = ReplayMemory(self.config, self.log_dir)

        has_pre_trained_model_flag = self.checkpoint_dir + "/{}.best.model.index".format(config.model_name)
        if os.path.exists(has_pre_trained_model_flag):
            self.has_pre_trained_model = True
        else:
            self.has_pre_trained_model = False

        self.global_step = tf.train.get_or_create_global_step()

        self.total_loss = None
        self.total_q = None
        self.update_count = None
        self.step = 0

        self.w = None
        self.w_input = None
        self.t_w = None
        self.t_w_input = None

        self.value = None
        self.advantage = None
        self.t_value = None
        self.t_advantage = None

        self.q = None
        self.target_q = None
        self.target_q_idx = None
        self.target_q_with_idx = None

        self.action = None
        self.q_action = None
        self.target_q_t = None
        self.learning_rate_step = None
        self.learning_rate_op = None

        self.summary = None
        self.loss = None

        tensorboard_log = os.path.join(self.log_dir, "tensorboard")
        if not os.path.exists(tensorboard_log): os.makedirs(tensorboard_log)
        self.summary_writer = tf.summary.FileWriter(tensorboard_log, sess.graph)
        self.summary_placeholders = None
        self.summary_ops = None

        # build model graph
        self.create_placeholders()
        self.build_dqn()
        self.saver = tf.train.Saver(max_to_keep=10)

        if self.has_pre_trained_model:
            self.load_model()

    def create_placeholders(self):
        if self.config.cnn_format == 'NHWC':
            self.s_t = tf.placeholder('float32',
                                      [None, self.config.screen_height, self.config.screen_width,
                                       self.config.history_length],
                                      name='s_t')

            self.target_s_t = tf.placeholder('float32',
                                             [None, self.config.screen_height, self.config.screen_width,
                                              self.config.history_length],
                                             name='target_s_t')
        else:
            self.s_t = tf.placeholder('float32',
                                      [None, self.config.history_length, self.config.screen_height,
                                       self.config.screen_width],
                                      name='s_t')

            self.target_s_t = tf.placeholder('float32',
                                             [None, self.config.history_length, self.config.screen_height,
                                              self.config.screen_width],
                                             name='target_s_t')

        self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
        self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
        self.action = tf.placeholder('int64', [None], name='action')
        self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')

    def build_dqn(self):
        self.w = {}
        self.t_w = {}

        initializer = tf.contrib.layers.xavier_initializer()
        # initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu
        conv2d = Agent.conv2d
        dense = Agent.dense
        cnn_format = self.config.cnn_format

        # graph handler network
        with tf.variable_scope('prediction'):
            x = self.s_t
            strides = [4, 2, 1]
            kernels = [8, 4, 3]
            filters = [32, 64, 64]
            for i in range(1, 3):
                with tf.variable_scope('cnn_level_%d' % i):
                    x, self.w['l{}_w'.format(i)], self.w['l{}_b'.format(i)] \
                        = conv2d(x, filters[i - 1], [kernels[i - 1], kernels[i - 1]], [strides[i - 1], strides[i - 1]],
                                 initializer, activation_fn, cnn_format, name='cnn_l{}'.format(i))

            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, reduce(lambda a, b: a * b, shape[1:])])

            if self.config.dueling:
                value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
                    dense(x, 512, activation_fn=activation_fn, name='value_hid')
                self.value, self.w['val_w_out'], self.w['val_w_b'] = \
                    dense(value_hid, 1, name='value_out')

                adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
                    dense(x, 512, activation_fn=activation_fn, name='adv_hid')
                self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
                    dense(adv_hid, self.env.action_size, name='adv_out')

                # Average Dueling
                self.q = self.value + (self.advantage -
                                       tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
            else:
                self.value, self.w['l4_w'], self.w['l4_b'] = dense(x, 512, activation_fn=activation_fn, name='dense_l4')
                self.q, self.w['q_w'], self.w['q_b'] = dense(self.value, self.env.action_size, name='q_value')

            self.q_action = tf.argmax(self.q, axis=1)

            avg_q = tf.reduce_mean(self.q, 0)
            for idx in range(self.env.action_size):
                tf.summary.histogram('q/%s' % idx, avg_q[idx])
            self.summary = tf.summary.merge_all()

        # target network
        with tf.variable_scope('target'):
            y = self.target_s_t
            strides = [4, 2, 1]
            kernels = [8, 4, 3]
            filters = [32, 64, 64]
            for i in range(1, 3):
                with tf.variable_scope('target_cnn_level_%d' % i):
                    y, self.t_w['l{}_w'.format(i)], self.t_w['l{}_b'.format(i)] \
                        = conv2d(y, filters[i - 1], [kernels[i - 1], kernels[i - 1]],
                                 [strides[i - 1], strides[i - 1]],
                                 initializer, activation_fn, cnn_format, name='cnn_target_l{}'.format(i))

            shape = y.get_shape().as_list()
            y = tf.reshape(y, [-1, reduce(lambda a, b: a * b, shape[1:])])

            if self.config.dueling:
                t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] = \
                    dense(y, 512, activation_fn=activation_fn, name='target_value_hid')
                self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
                    dense(t_value_hid, 1, name='target_value_out')

                t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] = \
                    dense(y, 512, activation_fn=activation_fn, name='target_adv_hid')
                self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
                    dense(t_adv_hid, self.env.action_size, name='target_adv_out')

                # Average Dueling
                self.target_q = self.t_value + (self.t_advantage -
                                                tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
            else:
                self.t_value, self.t_w['l4_w'], self.t_w['l4_b'] = \
                    dense(y, 512, activation_fn=activation_fn, name='target_l4')
                self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
                    dense(self.t_value, self.env.action_size, name='target_q')

            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w[name] = self.t_w[name].assign(self.t_w_input[name])

        # optimizer
        with tf.variable_scope('optimizer'):
            action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

            delta = self.target_q_t - q_acted
            self.loss = tf.reduce_mean(Agent.clipped_error(delta), name='loss')
            self.learning_rate_op = tf.maximum(self.config.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   self.config.learning_rate,
                                                   self.learning_rate_step,
                                                   self.config.learning_rate_decay_step,
                                                   self.config.learning_rate_decay,
                                                   staircase=True))

            self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95,
                                                   epsilon=0.01).minimize(self.loss)

        with tf.variable_scope('summary'):
            self.summary_placeholders = {}
            self.summary_ops = {}

            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q',
                                   'episode.max_reward', 'episode.min_reward', 'episode.avg_reward',
                                   'episode.num_of_game', 'training.learning_rate']
            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar("%s/%s" % (self.config.env_name, tag),
                                                          self.summary_placeholders[tag])

            histogram_summary_tags = ['episode.rewards', 'episode.actions']
            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])

        tf.global_variables_initializer().run()

        if self.has_pre_trained_model:
            self.load_model()

    def train(self):
        start_step = self.global_step.eval()
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []

        screen, reward, action, terminal = self.env.new_game()

        for _ in range(self.config.history_length):
            self.history.add(screen)

        for self.step in tqdm(range(start_step, self.config.max_step), ncols=70, initial=start_step):
            if self.step == self.config.learn_start_step:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            # 1. predict
            action = self.predict(self.history.get())

            # 2. act
            screen, reward, terminal = self.env.act(action)

            # 3. observe
            self.observe(screen, reward, action, terminal)

            if terminal:
                screen, reward, action, terminal = self.env.new_game()
                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward

            if self.step >= self.config.learn_start_step:
                if (self.step - self.config.learn_start_step) % self.config.test_step == self.config.test_step - 1:
                    avg_reward = total_reward / self.config.test_step
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count

                    max_ep_reward = np.max(ep_rewards)
                    min_ep_reward = np.min(ep_rewards)
                    avg_ep_reward = np.mean(ep_rewards)

                    print('\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, '
                          'min_ep_r: %.4f, # game: %d' % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward,
                                                          min_ep_reward, num_game))

                    if max_avg_ep_reward * 0.9 <= avg_ep_reward and \
                            self.step % self.config.save_step == self.config.save_step - 1:
                        self.save_model(name=None, step=self.step + 1)
                        max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

                    self.inject_summary(
                        {'average.reward': avg_reward,
                         'average.loss': avg_loss,
                         'average.q': avg_q,
                         'episode.max_reward': max_ep_reward,
                         'episode.min_reward': min_ep_reward,
                         'episode.avg_reward': avg_ep_reward,
                         'episode.num_of_game': num_game,
                         'episode.rewards': ep_rewards,
                         'episode.actions': actions,
                         'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step})
                         })

                    # after test, reset the game statistics
                    num_game, self.update_count, ep_reward = 0, 0, 0.
                    total_reward, self.total_loss, self.total_q = 0., 0., 0.
                    ep_rewards, actions = [], []

    def predict(self, s_t, epsilon=None):
        if epsilon is None:
            if self.step <= self.config.learn_start_step:
                # procedure for warming up
                epsilon = self.config.epsilon_start

            elif self.config.learn_start_step < self.step <= self.config.epsilon_end_step:
                epsilon_by_step = (self.config.epsilon_start - self.config.epsilon_end) * \
                                  (self.step - self.config.learn_start_step) / \
                                  (self.config.epsilon_end_step - self.config.learn_start_step)
                epsilon = self.config.epsilon_end + epsilon_by_step

            else:
                epsilon = self.config.epsilon_end

        if random.random() <= epsilon:
            # procedure for warming up
            action = random.randrange(self.env.action_size)
        else:
            action = self.q_action.eval({self.s_t: [s_t]}, session=self.sess)[0]

        return action

    def observe(self, screen, reward, action, terminal):
        reward = max(self.config.max_punishment, min(self.config.max_reward, reward))

        self.history.add(screen)
        self.memory.add(screen, reward, action, terminal)

        if self.step > self.config.learn_start_step:
            if self.step % self.config.train_step == 0:
                self.q_learning_mini_batch()

            if self.step % self.config.target_q_update_step == self.config.target_q_update_step - 1:
                self.update_target_q_network()

    def q_learning_mini_batch(self):
        if self.memory.count < self.config.history_length:
            return
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

        if self.config.double_q:
            # Double Q-learning
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})

            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
                self.target_s_t: s_t_plus_1,
                self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
            })
            target_q_t = (1. - terminal) * self.config.discount * q_t_plus_1_with_pred_action + reward

        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

            terminal = np.array(terminal) + 0.
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = (1. - terminal) * self.config.discount * max_q_t_plus_1 + reward

        _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.summary], {
            self.target_q_t: target_q_t,
            self.action: action,
            self.s_t: s_t,
            self.learning_rate_step: self.step,
        })

        self.summary_writer.add_summary(summary_str, self.step)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w[name].eval({self.t_w_input[name]: self.w[name].eval()})

    def save_weight_to_pkl(self):
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)

        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

    def load_weight_from_pkl(self):
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
                self.w[name] = self.w[name].assign(self.w_input[name])
                self.w[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

        self.update_target_q_network()

    def inject_summary(self, tag_dict):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
            self.summary_writer.add_summary(summary_str, self.step)

        self.summary_writer.flush()

    def play(self, n_episode=100, epsilon=None):
        if epsilon is None:
            epsilon = self.config.epsilon_end

        test_history = History(self.config)

        if not self.config.display:
            gym_dir = '/tmp/%s-%s' % (self.config.env_name, get_time())
            self.env.env.monitor.start(gym_dir)

        best_reward, best_idx = 0, 0
        for idx in range(n_episode):
            screen, reward, action, terminal = self.env.new_game()
            current_reward = 0

            for _ in range(self.config.history_length):
                test_history.add(screen)

            terminal = False
            while terminal is not True:
                action = self.predict(test_history.get(), epsilon)
                screen, reward, terminal = self.env.act(action)
                test_history.add(screen)
                current_reward += reward

            if current_reward > best_reward:
                best_reward = current_reward
                best_idx = idx

        print("=" * 30)
        print(">>>>>   [%d] Best reward : %d" % (best_idx, best_reward))
        print("=" * 30)

        if not self.config.display:
            self.env.env.monitor.close()

    @staticmethod
    def conv2d(_input, output_dim, kernel_size, stride,
               initializer=tf.contrib.layers.xavier_initializer(),
               activation_fn=tf.nn.relu, data_format='NHWC',
               padding='SAME', name='conv2d'):

        with tf.variable_scope(name):
            if data_format == 'NCHW':
                stride = [1, 1, stride[0], stride[1]]
                kernel_shape = [kernel_size[0], kernel_size[1], _input.get_shape()[1], output_dim]
            else:
                # data_format 'NHWC'
                stride = [1, stride[0], stride[1], 1]
                kernel_shape = [kernel_size[0], kernel_size[1], _input.get_shape()[-1], output_dim]

            kernel = tf.get_variable('kernel', kernel_shape, tf.float32, initializer=initializer)
            conv = tf.nn.conv2d(_input, kernel, stride, padding, data_format=data_format)

            b = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer())
            out = tf.nn.bias_add(conv, b, data_format)

            if activation_fn is not None:
                out = activation_fn(out)

        return out, kernel, b

    @staticmethod
    def clipped_error(_input):
        # Huber loss
        return tf.where(tf.abs(_input) < 1.0, 0.5 * tf.square(_input), tf.abs(_input) - 0.5)

    @staticmethod
    def dense(_input, output_dim, stddev=0.02, activation_fn=None, name='linear'):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [_input.get_shape()[1], output_dim], tf.float32,
                                tf.random_normal_initializer(stddev=stddev))
            b = tf.get_variable('bias', [output_dim],
                                initializer=tf.constant_initializer())
            out = tf.nn.xw_plus_b(_input, w, b)

            if activation_fn is not None:
                out = activation_fn(out)

        return out, w, b
