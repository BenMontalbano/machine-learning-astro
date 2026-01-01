import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Author: Ben Montalbano
# Purpose: The purpose of this script is to predict stable and unstable orbits of planets in a binary star system
# using randomly genreated planets and logistic regression.

# Define Gravitational Constant, speed of light, and step size
GC = 6.743E-11  # SI units
c = 299792458 # meters/second
TIMEINC = 3000  # secs

# Create Storage systems for data of bodies
class body:
    def __init__(self, name, mass, Xpos, Ypos, Zpos, Xvel, Yvel, Zvel):
        self.name = name
        self.mass = mass  # kg
        self.Xpos = Xpos  # m
        self.Ypos = Ypos  # m
        self.Zpos = Zpos
        self.Xvel = Xvel  # m/s
        self.Yvel = Yvel  # m/s
        self.Zvel = Zvel  # m/s


#Input Data for Bodies
BinaryPlus = body('BinaryPlus', 2*1.989E30, -2.3E10, 0, 0, 0, 25100 , 0)  
BinaryMinus = body('BinaryMinus', 1.989E30, 4.7E10, 0, 0, 0, -50300 , 0) 
BinaryBodies=[]
InitialDistance=[]
Planets=list(map(chr, range(ord('a'), ord('f')+1)))
 Planets = []
letters='abcdefghijklmnopqrstuvwxyz'
for first in letters:
    for second in letters:
        Planets.append(first + second)
        if len(Planets) == 100:
            break
    if len(Planets) == 100:
        break
for Planet in Planets:
    Mass=random.uniform(1E22,2E27)
    Orbital_Distance=random.uniform(8E10,4E11)
    phi=random.uniform(0,2*np.pi)
    theta=random.uniform(-np.pi/6,np.pi/6)
    Orbital_XVel=np.sqrt(GC*(BinaryPlus.mass + BinaryMinus.mass)/Orbital_Distance)*np.cos(theta)*np.cos(phi)
    Orbital_YVel=np.sqrt(GC*(BinaryPlus.mass + BinaryMinus.mass)/Orbital_Distance)*np.cos(theta)*np.sin(phi)
    Orbital_ZVel=np.sqrt(GC*(BinaryPlus.mass + BinaryMinus.mass)/Orbital_Distance)*np.sin(theta)
    planet=body(Planet,Mass,Orbital_Distance*np.sin(phi),Orbital_Distance*np.cos(phi),0,Orbital_XVel,Orbital_YVel, Orbital_ZVel)
    BinaryBodies.append(planet)
    InitialDistance.append(Orbital_Distance)

    
# Create List of Bodies
Bodies=[BinaryPlus, BinaryMinus]
Bodies += BinaryBodies
colors = ['white','white', 'red','green', 'yellow','blue','orange','purple']
Num_Bodies = len(Bodies)

# Create Arrays for Newobject with Distances to each Body
def Distances(Newobject):
    
    DistanceArray = np.zeros((Num_Bodies, 1), dtype=np.float64)
    i = 0
    for Body in Bodies:
        RadSquare = ((Newobject.Xpos - Body.Xpos) ** 2 +
                     (Newobject.Ypos - Body.Ypos) ** 2 +
                     (Newobject.Zpos - Body.Zpos) ** 2)
        DistanceArray[i] = RadSquare
        i += 1
    return DistanceArray

# Find Gravitational Force by each body
def Gravity(Newobject):
    DistanceArray = Distances(Newobject)
    i = 0
    GForce = np.zeros((3, 1), dtype=np.float64)

    for Body in Bodies:
        if Body == Newobject:
            i += 1
            continue
        Xgrav = -((GC * Body.mass * Newobject.mass) / DistanceArray[i]) * \
                ((Newobject.Xpos - Body.Xpos) / np.sqrt(DistanceArray[i]))
        Ygrav = -((GC * Body.mass * Newobject.mass) / DistanceArray[i]) * \
                ((Newobject.Ypos - Body.Ypos) / np.sqrt(DistanceArray[i]))
        Zgrav = -((GC * Body.mass * Newobject.mass) / DistanceArray[i]) * \
                ((Newobject.Zpos - Body.Zpos) / np.sqrt(DistanceArray[i]))
        GForce[0] += Xgrav
        GForce[1] += Ygrav
        GForce[2] += Zgrav
        i += 1

    return GForce

# Find Motion of Newobject in a Timestep
def Motion(Newobject):
    Gforce = Gravity(Newobject).flatten()
    Xacc = Gforce[0] / Newobject.mass
    Yacc = Gforce[1] / Newobject.mass
    Zacc = Gforce[2] / Newobject.mass

    Newobject.Xvel += Xacc * TIMEINC
    Newobject.Yvel += Yacc * TIMEINC
    Newobject.Zvel += Zacc * TIMEINC

    Newobject.Xpos += Newobject.Xvel * TIMEINC
    Newobject.Ypos += Newobject.Yvel * TIMEINC
    Newobject.Zpos += Newobject.Zvel * TIMEINC

    return [Newobject.Xpos, Newobject.Ypos, Newobject.Zpos,
            Newobject.Xvel, Newobject.Yvel, Newobject.Zvel]

def Ending_Place(Newobject):
    DistanceArrayTest = np.zeros((3, 1), dtype=np.float64)
    i = 0
    for Body in Test_Group:
        RadSquare = ((Newobject.Xpos - Body.Xpos) ** 2 +
                     (Newobject.Ypos - Body.Ypos) ** 2 +
                     (Newobject.Zpos - Body.Zpos) ** 2)
        DistanceArrayTest[i] = RadSquare
        i += 1
    return DistanceArrayTest

def Stability_Tester(planet, duration=1E8, timeinc=3000):
    timesteps = int(duration//timeinc)
    for step in range(timesteps):
        Motion(planet)
        Ending_Place(planet)
        DistanceArrayTest=Ending_Place(planet)
        if np.max(DistanceArrayTest)>2E23:
            result='Unstable'
        else:
            result='Stable'
    return result

def extract_features(body):
    return [body.mass, body.Xpos, body.Ypos, body.Zpos,
            body.Xvel, body.Yvel, body.Zvel]

Stability_Array=[]
for body in BinaryBodies:
    result=Stability_Tester(body)
    Stability_Array.append(result)

Testers=[extract_features(body) for body in BinaryBodies]
X_train, X_test, y_train, y_test = train_test_split(Testers, Stability_Array, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

for i,result in enumerate(Stability_Array):
    if result=='Stable':
        Planet_Name= BinaryBodies[i].name
        Planet_Mass= "{:.2e}".format(BinaryBodies[i].mass)
        Planet_Orbit= "{:.2e}".format(BinaryBodies[i].Xpos)
        print("Stable Planet:")
        print(f"Planet Name-{Planet_Name}")
        print(f"Mass-{Planet_Mass}kg")
        print(f"Orbital_Distance-{Planet_Orbit}m")
        
    else:
        Planet_Name= BinaryBodies[i].name
        Planet_Mass= "{:.2e}".format(BinaryBodies[i].mass)
        Planet_Orbit= "{:.2e}".format(InitialDistance[i])
        print("Unstable Planet:")
        print(f"Planet Name:{Planet_Name}")
        print(f"Mass:{Planet_Mass}kg")
        print(f"Orbital_Distance:{Planet_Orbit}m")

print("Accuracy:", accuracy_score(y_test, y_pred))
print(Stability_Array)

"""
# Setup for Simulation
Duration = 100000000
Timesteps = Duration // TIMEINC
XPaths = np.zeros((Num_Bodies, Timesteps), dtype=np.float64)
YPaths = np.zeros((Num_Bodies, Timesteps), dtype=np.float64)
ZPaths = np.zeros((Num_Bodies, Timesteps), dtype=np.float64)

# Find X, Y, and Z paths for each body
for j in range(Timesteps):
    Xpoints = np.zeros(Num_Bodies, dtype=np.float64)
    Ypoints = np.zeros(Num_Bodies, dtype=np.float64)
    Zpoints = np.zeros(Num_Bodies, dtype=np.float64)

    for i, ClassObject in enumerate(Bodies):
        new_pos = Motion(ClassObject)
        ClassObject.Xpos, ClassObject.Ypos, ClassObject.Zpos = new_pos[0:3]
        ClassObject.Xvel, ClassObject.Yvel, ClassObject.Zvel = new_pos[3:6]

        Xpoints[i] = new_pos[0]
        Ypoints[i] = new_pos[1]
        Zpoints[i] = new_pos[2]

    XPaths[:, j] = Xpoints
    YPaths[:, j] = Ypoints
    ZPaths[:, j] = Zpoints


# Create Figure for Solar System animation
fig = plt.figure()
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black') 
ax.set_xlim(-3E11, 3E11)
ax.set_ylim(-3E11, 3E11)
ax.set_zlim(-3E11, 3E11)
ax.set_axis_off() 
ax.grid(False)


Paths = [ax.plot([], [], [], color=colors[i % len(colors)])[0] for i in range(Num_Bodies)]
Points = [
    ax.scatter([], [], [], 'o', color=colors[i % len(colors)], s = 250 if i == 0 else (150 if i == 1 else 60))
    for i in range(Num_Bodies)
]

# Place Bodies
def Place_Planets():
    for Path, Point in zip(Paths, Points):
        empty_array = np.array([])
        Path.set_data(empty_array, empty_array)
        Path.set_3d_properties(empty_array)
        Point._offsets3d = ([], [], [])
    return Paths + Points

# Update Bodies
def Next_Frame(frame):
    ax.view_init(elev=70, azim=45)
    for i in range(Num_Bodies):
        Paths[i].set_data(XPaths[i, :frame], YPaths[i, :frame])
        Paths[i].set_3d_properties(ZPaths[i, :frame])
        Points[i]._offsets3d = ([XPaths[i, frame]], [YPaths[i, frame]], [ZPaths[i, frame]])
    return Paths + Points

# Animate System
ani = animation.FuncAnimation(fig, Next_Frame, frames=range(0, Timesteps, 30),
                              init_func=Place_Planets, blit=False, interval=1)

plt.show()"""
