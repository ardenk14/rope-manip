from panda_rope_env import PandaRopeEnv, rope_data_t
import numpy as np
import random

COLLECT_RATE = 5 #collect a sample every 10 timesteps

def collect_data_sample(fname):
    env = PandaRopeEnv(gui=True)
    env.load_rope(ESt=1.0, DSt=1.0, BSt=1.0, Rp=1.0, mass=0.5) #TODO: randomize this
    env._reset_joints()
    env.close_gripper()

    ok = True
    #let rope fall to initial position
    for _ in range(100):
        ok *= env.stepSimulation()
    
    if not ok:
        return False, None
    
    data = np.empty(0, dtype=rope_data_t)

    for push_idx in range(10):
        rope_pos = random.choice(env.get_deform_points())
        if np.linalg.norm(rope_pos[:2]) < 0.70 and np.linalg.norm(rope_pos[:2]) > 0.25: #in robot reach
            rope_pos[2] = 0.003

            magnitude = 0.1
            theta = np.random.uniform(-3.14, 3.14)
            direction = np.array([np.cos(theta), np.sin(theta), 0])

            # move above 
            env.close_gripper()
            ok, move_data = env.move_ik_data(rope_pos + magnitude*direction + [0,0,0.05], collection_rate=COLLECT_RATE)
            if ok:
                data = np.concatenate((data, move_data))
            else:
                return False, None

            env.close_gripper()
            ok, move_data = env.move_ik_data(rope_pos + magnitude*direction, collection_rate=COLLECT_RATE)
            if ok:
                data = np.concatenate((data, move_data))
            else:
                return False, None

            # push
            env.close_gripper()
            ok, move_data = env.move_ik_data(rope_pos - magnitude*direction, collection_rate=COLLECT_RATE)
            if ok:
                data = np.concatenate((data, move_data))
            else:
                return False, None
            
            env.close_gripper()
            ok, move_data = env.move_ik_data(rope_pos - magnitude*direction + [0,0,0.05], collection_rate=COLLECT_RATE)
            if ok:
                data = np.concatenate((data, move_data))
            else:
                return False, None
            
            print(data.shape)

    #TODO: save data

if __name__ == '__main__':
    collect_data_sample('test')