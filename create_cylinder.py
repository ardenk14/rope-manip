import gmsh
from math import pi

gmsh.initialize()

gmsh.model.add("cylinder")

gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.4)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.4)
gmsh.option.setNumber("Mesh.MeshSizeFactor", 0.1)

x, y, z = 0,0,0
dx, dy, dz = 0,0,1
radius = 0.1
gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, radius, tag = 1, angle = 2*pi)

gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(3, [1], 1)
gmsh.model.setPhysicalName(3, 1, "The volume")

gmsh.model.mesh.generate(3)
gmsh.write("cylinder.vtk")

gmsh.fltk.run()
gmsh.finalize()

