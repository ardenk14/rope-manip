from panda_env import PandaEnv
from panda_pushing_env import PandaImageSpacePushingEnv
from panda_rope_env import PandaRopeEnv
import pybullet as pb
import time


if __name__ == '__main__':
    # TODO: Create rope manipulation environment based on the two Pandas environments we have
    # TODO: Get arm to pick and place rope with lots of motion while in its hand
    # TODO: Collect images of rope and manipulation actions/proprioception and train autoencoder
    # TODO: Train dynamics model

    #pb.connect(pb.GUI)
    #env = PandaImageSpacePushingEnv()#PandaEnv()
    env = PandaRopeEnv(gui=True)
    env.load_rope()
    #env.reset()
    env._reset_joints()
    joint_angles = env.get_joint_angles()
    env.set_joint_angles(joint_angles)
    while True:
        img = env.get_image()
        env.stepSimulation()
        print(env.get_ee_pos())
    time.sleep(100)