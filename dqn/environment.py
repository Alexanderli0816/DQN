import gym
from .utils import rgb2gray, imresize


class Environment(object):
    def __init__(self, config):
        self.env = gym.make(config.env_name)

        screen_width, screen_height = config.screen_width, config.screen_height

        self.display = config.display
        self.dims = (screen_width, screen_height)

        self._screen = self.env.reset()
        self.reward = 0
        self.terminal = True

    def new_game(self):
        self._screen = self.env.reset()
        self.render()
        return self.screen, 0, 0, self.terminal

    def step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)

    def random_step(self):
        action = self.env.action_space.sample()
        self.step(action)

    @property
    def screen(self):
        return imresize(rgb2gray(self._screen) / 255., self.dims)
        # return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def lives(self):
        return self.env.ale.lives()

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    def render(self):
        if self.display:
            self.env.render()


class GymEnvironment(Environment):
    def __init__(self, config):
        super(GymEnvironment, self).__init__(config)
        self.config = config

    def act(self, action):
        accumulated_reward = 0
        start_lives = self.lives

        self.step(action)
        if start_lives > self.lives and self.terminal:
            accumulated_reward -= self.config.max_punishment
        self.render()
        return self.state
