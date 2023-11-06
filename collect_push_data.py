from panda_rope_env import PandaRopeEnv, rope_data_t
import numpy as np
import random
import multiprocessing as mp

COLLECT_RATE = 10 #collect a sample every 10 timesteps

def collect_data_sample(fname):
    print('---------- Random: ', random.random())
    env = PandaRopeEnv(gui=True)
    ESt=random.uniform(0.5, 2.5)
    DSt=random.uniform(0.5, 2.5)
    BSt=random.uniform(0.05, 5.0)
    Rp=random.uniform(0.5, 2.5)
    mass=random.uniform(0.1, 2.0)

    print(f'Started with Est:{ESt}, DSt:{DSt}, BSt:{BSt}, Rp:{Rp}, mass:{mass}')
    env.load_rope(ESt=ESt, DSt=DSt, BSt=BSt, Rp=Rp, mass=mass, angle=random.uniform(np.pi/30, -np.pi/10))
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
            env.stepSimulation()
            ok, move_data = env.move_ik_data(rope_pos + magnitude*direction + [0,0,0.05], collection_rate=COLLECT_RATE)
            if ok:
                # print(data.shape, move_data.shape)
                data = np.concatenate((data, move_data))
            else:
                np.savez_compressed(file=fname, ESt=ESt, DSt= DSt, BSt= BSt, Rp= Rp, mass= mass, data= data)
                return False

            env.close_gripper()
            env.stepSimulation()
            ok, move_data = env.move_ik_data(rope_pos + magnitude*direction, collection_rate=COLLECT_RATE)
            if ok:
                # print(data.shape, move_data.shape)
                data = np.concatenate((data, move_data))
            else:
                np.savez_compressed(file=fname, ESt=ESt, DSt= DSt, BSt= BSt, Rp= Rp, mass= mass, data= data)
                return False

            # push
            env.close_gripper()
            env.stepSimulation()
            ok, move_data = env.move_ik_data(rope_pos - magnitude*direction, collection_rate=COLLECT_RATE)
            if ok:
                # print(data.shape, move_data.shape)
                data = np.concatenate((data, move_data))
            else:
                np.savez_compressed(file=fname, ESt=ESt, DSt= DSt, BSt= BSt, Rp= Rp, mass= mass, data= data)
                return False
            
            env.close_gripper()
            env.stepSimulation()
            ok, move_data = env.move_ik_data(rope_pos - magnitude*direction + [0,0,0.05], collection_rate=COLLECT_RATE)
            if ok:
                # print(data.shape, move_data.shape)
                data = np.concatenate((data, move_data))
            else:
                np.savez_compressed(file=fname, ESt=ESt, DSt= DSt, BSt= BSt, Rp= Rp, mass= mass, data= data)
                return False
    print(data.shape)
            
    np.savez_compressed(file=fname, ESt=ESt, DSt= DSt, BSt= BSt, Rp= Rp, mass= mass, data= data)

if __name__ == '__main__':
    # collect samples
    nsamples = 2
    with mp.Pool(2) as pool:
        filenames = ['test{}'.format(i) for i in range(nsamples)]
        pool.map(collect_data_sample, filenames)

    # read sample
    sample = np.load('test0.npz')
    print('----- Read Sample test0.npz ---------')
    print('Elastic Stiffness:', sample['ESt'])
    print('Data Shape:', sample['data'].shape, 'dtype', sample['data'].dtype)