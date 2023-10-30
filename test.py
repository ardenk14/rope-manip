from panda_rope_env import PandaRopeEnv
import pybullet as pb
import time
import random


if __name__ == '__main__':
    # TODO: Create rope manipulation environment based on the two Pandas environments we have
    # TODO: Get arm to pick and place rope with lots of motion while in its hand
    # TODO: Collect images of rope and manipulation actions/proprioception and train autoencoder
    # TODO: Train dynamics model

    env = PandaRopeEnv(gui=True)
    env.load_rope(ESt=1.0, DSt=1.0, BSt=1.0, Rp=1.0, mass=0.5)
    env._reset_joints()

    env.close_gripper()
    #let rope fall to initial position
    for _ in range(200):
        env.stepSimulation()


    for i in range(10):
        print(env.get_joint_angles())
        print(env.get_joint_angles().dtype)
        # env.push_rope()
        
    #     #open gripper
    #     print('push', i)
        # go to a random vertex on the rope
        # print('grabbing rope')
        # rope_pos = random.choice(env.get_deform_points())
        # print(rope_pos)
        # rope_pos[2] = 0.003
        # env.move_ik_blocking(rope_pos + [0,0,0.05], tol=0.02)
        # env.move_ik_blocking(rope_pos, tol = 0.02)

        # env.close_gripper()
        # for _ in range(50):
        #     env.stepSimulation()

        # # pick up and drop it
        # env.move_ik_blocking(rope_pos + [0.0, 0.0, 0.15], tol=0.02)
    time.sleep(3)