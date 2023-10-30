import pybullet as p
from pybullet_utils import bullet_client
import pybullet_data
import numpy as np
import math
import time
import random

W = 320
H = 240
rope_data_t = np.dtype([('img', np.uint8, (H,W,3)),
                        ('seg_img', np.int32, (H,W)),
                        ('joint_angles', np.float32, (7,)),
                        ('ee_pos', np.float32, (3,)),
                        ('action', np.float32, (3,))]) # action is cartesian delta

class PandaRopeEnv():
    def __init__(self, dt=0.1, gui=False) -> None:
        self.dt = dt

        if gui:
            self._p = bullet_client.BulletClient(connection_mode=p.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self._p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        self._p.setGravity(0,0,-9.81)
        # self._p.setTimeStep(self.dt)
        self._p.setRealTimeSimulation(0)

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # planeId = self._p.loadURDF("plane.urdf")
        tableId = self._p.loadURDF("table/table.urdf", (0.4, 0, -0.625), self._p.getQuaternionFromEuler((0, 0, 3.1415/2.0)))
        self.panda = self._p.loadURDF('franka_panda/panda.urdf', useFixedBase=True)

        self.panda_joint_indxs = [0, 1, 2, 3, 4, 5, 6]
        self.finger_joint_indxs = [7, 8, 9, 10, 11]
        # self.reset_joints = [0., 0., 0., 0., 0., 0., 0.]
        self.reset_joints = np.array([0., 0., 0., -np.pi * 0.5, 0., np.pi * 0.5, 0.])
        self.rest_state = np.concatenate([self.reset_joints, np.zeros(7, )])
        self.joint_limits = [
            np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),  # lower limit
            np.array([2.8973, 1.7628, 2.8973, 0.0698, 2.8973, 3.7525, 2.8973]),  # upper limit
        ]
        self.vel_limits = [
            -np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),  # lower limit
            np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),  # upper limit
        ]
        self.joint_torque_limits = [
            -25 * np.ones(7),  # lower limit
            25 * np.ones(7),  # upper limit
        ]

        # Camera Setup
        self.img_width = W
        self.img_height = H

        self.cam_fov = 54
        self.img_aspect = self.img_width / self.img_height
        self.dpth_near = 0.02
        self.dpth_far = 5

        self.view_matrix = self._p.computeViewMatrix([1.0, 0.0, 0.8], [0.6, 0, 0.4], [0, 0, 1])
        self.projection_matrix = self._p.computeProjectionMatrixFOV(self.cam_fov, self.img_aspect, self.dpth_near, self.dpth_far)

        # planeId = p.loadURDF("plane.urdf", [-3,0,0], self._p.getQuaternionFromEuler((0, 3.1415/2.0, 0)))
        planeId2 = p.loadURDF("plane.urdf", [0,0,-0.625])

    def load_rope(self, 
                  file_path='assets/objects/cyl_100_1568.vtk', 
                  mass=0.007,
                  ESt = 3.0,
                  DSt = 1.0,
                  BSt = 0.05,
                  Rp = 1.0):
        
        # Soft body parameters
        mass = 0.1
        scale = 0.012#0.018
        # scale = 0.035
        softBodyId = 0
        useBend = True
        
        cMargin = 0.00475
        friction = 1.0

        tex = p.loadTexture("uvmap.png")

        self.ropeId = p.loadSoftBody(file_path,
                                     mass=mass, 
                                     scale=scale, 
                                     basePosition=[0.35, 0.5, 0.3],
                                    baseOrientation=p.getQuaternionFromEuler([0, math.pi / 3, -math.pi/2]),
                                    useNeoHookean=0, 
                                    useBendingSprings=useBend, 
                                    useMassSpring=1,
                                    springElasticStiffness=ESt,
                                    springDampingStiffness=DSt, 
                                    springBendingStiffness=BSt, 
                                    repulsionStiffness=Rp,
                                    useSelfCollision=0,
                                    collisionMargin=cMargin, 
                                    frictionCoeff=friction, 
                                    useFaceContact=1)
        
        self._p.changeVisualShape(self.ropeId, -1, rgbaColor=[1,1,1,1], textureUniqueId=tex, flags=0)
    
    def get_image(self):
        # Get rgb images
        images = self._p.getCameraImage(self.img_width,
                        self.img_height,
                        self.view_matrix,
                        self.projection_matrix,
                        shadow=True,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_img = np.reshape(images[2], (self.img_height, self.img_width, 4))[:,:,:3]
        
        return rgb_img
    
    def get_img_seg(self):
        # Get rgb and segmentation images
        images = self._p.getCameraImage(self.img_width,
                        self.img_height,
                        self.view_matrix,
                        self.projection_matrix,
                        shadow=True,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_img = np.reshape(images[2], (self.img_height, self.img_width, 4))[:,:,:3]
        seg_img = np.reshape(images[4], (self.img_height, self.img_width))
        
        return rgb_img, seg_img
    
    def set_joint_angles(self, q):
        self._p.setJointMotorControlArray(self.panda, self.panda_joint_indxs, self._p.POSITION_CONTROL, targetPositions = q)

    def _reset_joints(self):
        for i, joint_indx in enumerate(self.panda_joint_indxs):
            joint_rest_value = self.reset_joints[i]
            self._p.resetJointState(self.panda, joint_indx, joint_rest_value, targetVelocity=0)

    def get_joint_angles(self):
        states = self._p.getJointStates(self.panda, self.panda_joint_indxs)
        q = np.array([state[0] for state in states])
        return q
    
    def get_joint_vel(self):
        states = self._p.getJointStates(self.panda, self.panda_joint_indxs)
        q = np.array([state[1] for state in states])
        return q
    
    def open_gripper(self):
        '''
        commands gripper to open, sim will need to be stepped
        '''
        self._p.setJointMotorControl2(self.panda, self.finger_joint_indxs[2], self._p.POSITION_CONTROL, targetPosition=0.04, maxVelocity=0.5)
        self._p.setJointMotorControl2(self.panda, self.finger_joint_indxs[3], self._p.POSITION_CONTROL, targetPosition=0.04, maxVelocity=0.5)

    def close_gripper(self):
        self._p.setJointMotorControl2(self.panda, self.finger_joint_indxs[2], self._p.POSITION_CONTROL, targetPosition=0.0, maxVelocity=0.5)
        self._p.setJointMotorControl2(self.panda, self.finger_joint_indxs[3], self._p.POSITION_CONTROL, targetPosition=0.0, maxVelocity=0.5)
    
    def move_ik(self, goal_pos, rpy = [3.14, 0, 1.57]):
        '''
        move to a position. Not 'blocking' ie doesn't step the simulation just sets commands
        args:
            - goal_pos: [x,y,z] of desired end-effector pose
        '''
        goal_config = self._p.calculateInverseKinematics(self.panda, 
                                                        self.finger_joint_indxs[-1], 
                                                        goal_pos,
                                                        self._p.getQuaternionFromEuler(rpy),
                                                        lowerLimits=self.joint_limits[0],
                                                        upperLimits=self.joint_limits[1],
                                                        # jointRanges=self.range_limits,
                                                        restPoses=self.rest_state,
                                                        maxNumIterations=50,
                                                        # jointDamping=self.joint_damping,
                                                        solver=self._p.IK_DLS)
        numJoints = self._p.getNumJoints(self.panda)
        j = 0
        for i in range(numJoints):
            jointInfo = self._p.getJointInfo(self.panda, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self._p.setJointMotorControl2(self.panda, i, self._p.POSITION_CONTROL, targetPosition=goal_config[qIndex-7])
                j += 1

        return np.linalg.norm(goal_pos - self.get_ee_pos())
    
    def move_ik_blocking(self, goal_pos, tol=0.01, nsteps=200):
        '''
        moves with ik but also steps simulation until goal is reached
        '''
        dist = 1000000
        i = 0
        while dist > tol and i<nsteps:
            dist = self.move_ik(goal_pos)
            if not self.stepSimulation():
                return False # false for rope explosion
            i += 1
        return True
    
    def move_ik_data(self, goal_pos, collection_rate = 10, tol=0.01, nsteps=200):
        '''
        collect data while doing an ik move
        '''
        dist = 1000000
        i = 0
        data = []
        last_pos = self.get_ee_pos()
        while dist > tol and i<nsteps:

            if i > 0 and i % collection_rate == 0:
                step_data = np.zeros(1, dtype=rope_data_t)
                img, seg_img = self.get_img_seg()
                pos = self.get_ee_pos()
                step_data['img'] = img
                step_data['seg_img'] = seg_img
                step_data['joint_angles'] = self.get_joint_angles()
                step_data['ee_pos'] = pos
                step_data['action'] = pos - last_pos
                last_pos = pos
                data.append(step_data)

            dist = self.move_ik(goal_pos)
            if not self.stepSimulation():
                return False, None# false for rope explosion
            i += 1

        return True, np.array(data).squeeze()

    def get_ee_pos(self):
        return np.array(self._p.getLinkState(self.panda, self.finger_joint_indxs[-1])[0])

    def stepSimulation(self):
        self._p.stepSimulation()
        if self.ropeId is not None and self.is_rope_exploded():
            return False
        return True

    def disconnect(self):
        self._p.disconnect()

    def get_deform_points(self):
        kwargs = {'flags': p.MESH_DATA_SIMULATION_MESH}
        n_verts, mesh_verts_pos = self._p.getMeshData(self.ropeId, **kwargs)
        mesh_verts_pos = np.array(mesh_verts_pos)
        return mesh_verts_pos
    
    def is_rope_exploded(self):
        points = self.get_deform_points()
        bbx_sides = np.max(points, axis=0) - np.min(points, axis=0) #lengh of bbx sides
        return np.any(bbx_sides > 1.5)

    def push_rope(self):
        rope_pos = random.choice(self.get_deform_points())
        print(rope_pos)
        if np.linalg.norm(rope_pos[:2]) > 0.75 or np.linalg.norm(rope_pos[:2]) < 0.2: #out of robot reach
            return
        rope_pos[2] = 0.003

        magnitude = 0.1

        theta = np.random.uniform(-3.14, 3.14)
        direction = np.array([np.cos(theta), np.sin(theta), 0])

        ### TEST ###
        # rope_pos = self.get_deform_points()[100]
        # rope_pos[2] = 0.003

        # theta = 3.14
        # direction = np.array([np.cos(theta), np.sin(theta), 0])
        ############
        ok = True

        ok *= self.move_ik_blocking(rope_pos + magnitude*direction + [0,0,0.05])
        ok *= self.move_ik_blocking(rope_pos + magnitude*direction)

        ok *= self.move_ik_blocking(rope_pos - magnitude*direction)
        ok *= self.move_ik_blocking(rope_pos - magnitude*direction + [0,0,0.05])

        if not ok:
            # rope has exploded
            return False
        return True
    


        
        

if __name__ == '__main__':
    env = PandaRopeEnv(gui=True)
    env._reset_joints()
    joint_angles = env.get_joint_angles()
    env.set_joint_angles(joint_angles)
    while True:
        img = env.get_image()
        env.stepSimulation()
        print(env.get_ee_pos())
