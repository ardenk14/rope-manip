# import pybullet as p
# import pybullet_data
# from pybullet_utils import bullet_client

# if __name__ == '__main__':
#     _p = bullet_client.BulletClient(connection_mode=p.GUI)
#     _p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
#     _p.setGravity(0,0,-9.81)

#     tableId = _p.loadURDF("assets/objects/table/table.urdf", (0, 0, 0), 
#                           _p.getQuaternionFromEuler((0, 0, 3.1415/2.0)))
    
#     # rope_id = p.loadSoftBody('./assets/objects/cable.obj',
#     #                         basePosition = [0,0,2],
#     #                         scale = 1.0,
#     #                         mass = .1,
#     #                         useNeoHookean=1,
#     #                         NeoHookeanMu = 180, 
#     #                         NeoHookeanLambda = 600, 
#     #                         NeoHookeanDamping = 0.01, 
#     #                         collisionMargin = 0.006, 
#     #                         useSelfCollision = 1, 
#     #                         frictionCoeff = 0.5,
#     #                         )
#     _p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     bunny2 = _p.loadURDF("torus_deform.urdf", [0,1,0.5], flags=p.URDF_USE_SELF_COLLISION)
#     # bunnyId = p.loadSoftBody("torus/torus_textured.obj", basePosition=[0,0,1], mass = 3, useNeoHookean = 1, NeoHookeanMu = 180, NeoHookeanLambda = 600, NeoHookeanDamping = 0.01, collisionMargin = 0.006, useSelfCollision = 1, frictionCoeff = 0.5, repulsionStiffness = 800)
#     # rope_id = p.loadSoftBody('./assets/objects/cable.obj',
#     #                         basePosition = [0,0,2],
#     #                         scale = 1.0,
#     #                         mass = .1,
#     #                         useNeoHookean = 0, 
#     #                         useBendingSprings=1,
#     #                         useMassSpring=1, 
#     #                         springElasticStiffness=40, 
#     #                         springDampingStiffness=.1, 
#     #                         springDampingAllDirections = 1, 
#     #                         useSelfCollision = 1, 
#     #                         frictionCoeff = .5, 
#     #                         useFaceContact = 0)
    
#     for i in range(10000):
#         _p.stepSimulation()


import pybullet as p
from time import sleep
import pybullet_data
import math

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.resetDebugVisualizerCamera(3,-420,-30,[0.3,0.9,-2])
p.setGravity(0, 0, -10)

#tex = p.loadTexture("uvmap.png")
planeId = p.loadURDF("plane.urdf", [0,0,-2])

#boxId = p.loadURDF("cube.urdf", [0,3,2],useMaximalCoordinates = True)

#bunnyId = p.loadSoftBody("cylinder.vtk", simFileName="cylinder.vtk", mass = 3, useNeoHookean = 1, NeoHookeanMu = 180, NeoHookeanLambda = 600, NeoHookeanDamping = 0.01, collisionMargin = 0.006, useSelfCollision = 1, frictionCoeff = 0.5, repulsionStiffness = 800)
#p.changeVisualShape(bunnyId, -1, rgbaColor=[1,1,1,1], textureUniqueId=tex, flags=0)

# Soft body parameters
mass = 0.007
scale = 0.018
# scale = 0.035
softBodyId = 0
useBend = True
ESt = 0.19
DSt = 0.0625
BSt = 0.05
Rp = 0.01
cMargin = 0.00475
friction = 1e99

softBodyId = p.loadSoftBody('assets/objects/cyl_100_1568.vtk', mass=mass, scale=scale, #, basePosition=state_object
                            baseOrientation=p.getQuaternionFromEuler([0, math.pi / 2, -math.pi/2]),
                            useNeoHookean=0, useBendingSprings=useBend, useMassSpring=1,
                            springElasticStiffness=ESt,
                            springDampingStiffness=DSt, springBendingStiffness=BSt, repulsionStiffness=Rp,
                            useSelfCollision=0,
                            collisionMargin=cMargin, frictionCoeff=friction, useFaceContact=0)

# bunny2 = p.loadURDF("torus_deform.urdf", [0,1,0.5], flags=p.URDF_USE_SELF_COLLISION)

# p.changeVisualShape(bunny2, -1, rgbaColor=[1,1,1,1], textureUniqueId=tex, flags=0)
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
p.setRealTimeSimulation(0)

while p.isConnected():
  p.stepSimulation()
  p.getCameraImage(320,200)
  p.setGravity(0,0,-10)
