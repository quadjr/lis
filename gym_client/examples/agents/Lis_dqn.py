# coding:utf-8
import argparse
from cnn_dqn_agent import CnnDqnAgent
import gym
from PIL import Image
import numpy as np

import prednet
import os
import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.functions.loss.mean_squared_error import mean_squared_error

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--log-file', '-l', default='reward.log', type=str,
                    help='reward log file name')
args = parser.parse_args()

agent = CnnDqnAgent()
agent_initialized = False
cycle_counter = 0
log_file = args.log_file
reward_sum = 0
depth_image_dim = 32 * 32
depth_image_count = 1
total_episode = 10000
episode_count = 1

xp = cuda.cupy if args.gpu >= 0 else np
net = prednet.PredNet(227, 227, [3,16])
model = L.Classifier(net, lossfun=mean_squared_error)
model.compute_accuracy = False
optimizer = optimizers.Adam()
optimizer.setup(model)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    print('Running on a GPU')
else:
    print('Running on a CPU')


if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('images'):
    os.makedirs('images')

def write_image(image, path):
    image *= 255
    image = image.transpose(1, 2, 0)
    image = image.astype(np.uint8)
    result = Image.fromarray(image)
    result.save(path)


x_batch = np.ndarray((1, 3, 227, 227), dtype=np.float32)
y_batch = np.ndarray((1, 3, 227, 227), dtype=np.float32)

frame_count = 0
loss = 0
while episode_count <= total_episode:
    if not agent_initialized:
        agent_initialized = True
        
        print ("initializing agent...")
        agent.agent_init(
            use_gpu=args.gpu,
            depth_image_dim=depth_image_dim * depth_image_count)

        env = gym.make('Lis-v2')

        observation = env.reset()  
        action = agent.agent_start(observation)  
        observation, reward, end_episode, _ = env.step(action)
        
        net.reset_state()
        x_batch[0] = np.asarray(observation["image"][0]).transpose(2, 0, 1)/255.0

        with open(log_file, 'w') as the_file:
            the_file.write('cycle, episode_reward_sum \n')
    else:
        cycle_counter += 1
        reward_sum += reward

        if end_episode:
            agent.agent_end(reward)
            
            action = agent.agent_start(observation)  # TODO
            observation, reward, end_episode, _ = env.step(action)
            
            net.reset_state()
            x_batch[0] = np.asarray(observation["image"][0]).transpose(2, 0, 1)/255.0
            print x_batch[0]
            
            with open(log_file, 'a') as the_file:
                the_file.write(str(cycle_counter) +
                               ',' + str(reward_sum) + '\n')
            reward_sum = 0
            frame_count = 0
            episode_count += 1

        else:
            action, eps, q_now, obs_array = agent.agent_step(reward, observation)
            agent.agent_step_update(reward, action, eps, q_now, obs_array)
            observation, reward, end_episode, _ = env.step(action)
            
            y_batch[0] = np.asarray(observation["image"][0]).transpose(2, 0, 1)/255.0
            loss += model(chainer.Variable(xp.asarray(x_batch)),
                          chainer.Variable(xp.asarray(y_batch)))
            
            if (frame_count + 1) % 10 == 0:
                model.zerograds()
                loss.backward()
                loss.unchain_backward()
                loss = 0
                optimizer.update()
                if args.gpu >= 0:model.to_cpu()
                write_image(x_batch[0].copy(), 'images/' + str(cycle_counter) + '_x.jpg')
                write_image(model.y.data[0].copy(), 'images/' + str(cycle_counter) + '_y.jpg')
                write_image(y_batch[0].copy(), 'images/' + str(cycle_counter) + '_z.jpg')
                if args.gpu >= 0:model.to_gpu()
                print('loss:' + str(float(model.loss.data)))

            if (cycle_counter%10000) == 0:
                print('save the model')
                serializers.save_npz('models/' + str(cycle_counter) + '.model', model)
                print('save the optimizer')
                serializers.save_npz('models/' + str(cycle_counter) + '.state', optimizer)

            x_batch[0] = y_batch[0]
            frame_count += 1
            

env.close()
