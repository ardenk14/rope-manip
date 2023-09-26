import pybullet as p
from pybullet_utils import bullet_client
import pybullet_data
import numpy as np
import math

class PandaRopeEnv():
    def __init__(self, dt=0.1, gui=False) -> None:
        self.dt = dt

        if gui:
            self._p = bullet_client.BulletClient(connection_mode=p.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self._p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        self._p.setGravity(0,0,-9.81)
        #self._p.setTimeStep(self.dt)
        self._p.setRealTimeSimulation(0)

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # planeId = self._p.loadURDF("plane.urdf")
        tableId = self._p.loadURDF("table/table.urdf", (0.5, 0, -0.625), self._p.getQuaternionFromEuler((0, 0, 3.1415/2.0)))
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
        self.img_width = 640
        self.img_height = 480
        # self.img_width = 1280
        # self.img_height = 1280

        self.cam_fov = 54
        self.img_aspect = self.img_width / self.img_height
        self.dpth_near = 0.02
        self.dpth_far = 5

        self.view_matrix = self._p.computeViewMatrix([1.0, 0.0, 0.8], [0.6, 0, 0.4], [0, 0, 1])
        self.projection_matrix = self._p.computeProjectionMatrixFOV(self.cam_fov, self.img_aspect, self.dpth_near, self.dpth_far)

        # planeId = p.loadURDF("plane.urdf", [-3,0,0], self._p.getQuaternionFromEuler((0, 3.1415/2.0, 0)))
        planeId2 = p.loadURDF("plane.urdf", [0,0,-0.625])

    def load_rope(self, file_path='assets/objects/cyl_100_1568.vtk', mass=0.007):
        # Soft body parameters
        mass = 0.1
        scale = 0.012#0.018
        # scale = 0.035
        softBodyId = 0
        useBend = True
        ESt = 3.0
        DSt = 1.0
        BSt = 0.05
        Rp = 1.0
        cMargin = 0.00475
        friction = 1.0

        tex = p.loadTexture("uvmap.png")

        self.ropeId = p.loadSoftBody(file_path,
                                     mass=mass, 
                                     scale=scale, 
                                     basePosition=[0.6, 0.5, 0.44],
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
        # Get rgb, depth, and segmentation images
        images = self._p.getCameraImage(self.img_width,
                        self.img_height,
                        self.view_matrix,
                        self.projection_matrix,
                        shadow=True,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_img = np.reshape(images[2], (self.img_height, self.img_width, 4))[:,:,:3]
        
        return rgb_img
    
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
    
    def move_ik(self, goal_pos):
        '''
        move to a position. Not 'blocking' ie doesn't step the simulation just sets commands
        args:
            - goal_pos: [x,y,z] of desired end-effector pose
        '''
        goal_config = self._p.calculateInverseKinematics(self.panda, 
                                                        self.finger_joint_indxs[0], 
                                                        goal_pos,
                                                        self._p.getQuaternionFromEuler((0, 1.57, 0)),
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
                self._p.setJointMotorControl2(self.panda, i, self._p.POSITION_CONTROL, targetPosition=goal_config[qIndex-7], maxVelocity=0.5)
                j += 1

        return np.linalg.norm(goal_pos - self.get_ee_pos())
    
    def move_ik_blocking(self, goal_pos, tol=0.01):
        '''
        moves with ik but also steps simulation until goal is reached
        '''
        dist = 1000000
        while dist > tol:
            dist = self.move_ik(goal_pos)
            self.stepSimulation()

    def get_ee_pos(self):
        return self._p.getLinkState(self.panda, self.finger_joint_indxs[0])[0]

    def stepSimulation(self):
        self._p.stepSimulation()

    def disconnect(self):
        self._p.disconnect()

if __name__ == '__main__':
    env = PandaRopeEnv(gui=True)
    env._reset_joints()
    joint_angles = env.get_joint_angles()
    env.set_joint_angles(joint_angles)
    while True:
        img = env.get_image()
        env.stepSimulation()
        print(env.get_ee_pos())
