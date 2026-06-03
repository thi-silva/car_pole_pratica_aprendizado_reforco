import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gymnasium as gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam


def OurModel(input_shape, action_space):
    X_input = Input(shape=input_shape)

    #Primeira camada oculta com 512 nós 
    X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X_input)
    #Segunda Camada oculta com 256 nós
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    #Terceira camada oculta com 64 nós
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    #Camda de saída com função de ativação linear
    X = Dense(int(action_space), activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

    model.summary()
    return model

class DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        #Defini aqui a quantidade máxima de 500 episódios, de acordo com a atividade
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = int(self.env.action_space.n)
        self.EPISODES = 500
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95    # taxa de desconto
        self.epsilon = 1.0  # taxa de exploração do modelo
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995 #taxa de decaimento ajustada para 0.995
        self.batch_size = 64
        self.train_start = 500

       
        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomização do batch 
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

       
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                #Fluxo do DQN para a próxima ação de maior valor
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Treinamento da rede neural
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)


    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)
            
    def run(self):
        for e in range(self.EPISODES):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:                   
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    if i >= 300:
                        print("Saving trained model as cartpole-dqn.keras")
                        self.save("cartpole-dqn.keras")
                        return
                self.replay()

    def test(self):
        self.load("cartpole-dqn.keras")
        env_test = gym.make('CartPole-v1', render_mode="human")
        for e in range(3): #Ajustado para rodar por mais três episódios após a etapa de treinamento. Somente para comprovar que está funcional em 3 episódios de exemplo
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                action = np.argmax(self.model.predict(state))
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break

if __name__ == "__main__":
    agent = DQNAgent()
    #agent.run()
    agent.test()