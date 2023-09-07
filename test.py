from panda_env import PandaEnv
from panda_pushing_env import PandaImageSpacePushingEnv
import pybullet as pb
import time


if __name__ == '__main__':
    # TODO: Create rope manipulation environment based on the two Pandas environments we have
    # TODO: Get arm to pick and place rope with lots of motion while in its hand
    # TODO: Collect images of rope and manipulation actions/proprioception and train autoencoder
    # TODO: Train dynamics model

    #pb.connect(pb.GUI)
    #env = PandaImageSpacePushingEnv()#PandaEnv()
    env = PandaImageSpacePushingEnv(render_non_push_motions=True,  
                                camera_heigh=800, 
                                camera_width=800,
                                grayscale=True,
                                done_at_goal=False)
    env.reset()
    for i in range(30):
        action_i = env.action_space.sample()
        state, reward, done, info = env.step(action_i)
    time.sleep(100)