import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torch.nn as nn

# Author: Ben Montalbano
# Purpose: The purpose of this script is to predict stable and unstable orbits of planets in a binary star system
# using randomly genreated planets and machine learning.

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
        if len(Planets) == 2000:
            break
    if len(Planets) == 2000:
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
Test_Group = [BinaryPlus, BinaryMinus, Bodies[0]]
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
    GForce = np.zeros(3, dtype=np.float64)

    for Body in [BinaryPlus, BinaryMinus]:
        dx = Newobject.Xpos - Body.Xpos
        dy = Newobject.Ypos - Body.Ypos
        dz = Newobject.Zpos - Body.Zpos

        r_sq = dx**2 + dy**2 + dz**2
        r = np.sqrt(r_sq)

        factor = -(GC * Body.mass * Newobject.mass) / (r_sq * r)

        GForce[0] += factor * dx
        GForce[1] += factor * dy
        GForce[2] += factor * dz

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

# Locate last location of Bodies
def Ending_Place(Newobject):
    DistanceArrayTest = np.zeros((3, 1), dtype=np.float64)
    i = 0
    for Body in [BinaryMinus, BinaryPlus]:
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
        DistanceArrayTest=Ending_Place(planet)
        if np.max(DistanceArrayTest)>2E23:
            return 'Unstable'
    return 'Stable'

# Extract features of a body, not including mass
def extract_features(body):
    return [body.Xpos, body.Ypos, body.Zpos,
            body.Xvel, body.Yvel, body.Zvel]

X_raw=[extract_features(b) for b in BinaryBodies]

# Populate Array with stability results
Stability_Array=[]
for body in BinaryBodies:
    result=Stability_Tester(body)
    Stability_Array.append(result)

X = torch.tensor(X_raw, dtype=torch.float32)
y = torch.tensor([1.0 if s=='Stable' else 0.0 for s in Stability_Array], 
    dtype=torch.float32).unsqueeze(1)

# Normalize features
X_mean = X.mean(dim=0)
X_std = X.std(dim=0)
X = (X - X_mean) / (X_std + 1e-8)  # +1e-8 prevents divide by zero

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create Neural Network
class OrbitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(6,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
model =OrbitNet()

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

# Run Epochs
epochs=2000
losses=[]

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss=loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Print Accuracy
with torch.no_grad():
    y_pred_test = model(X_test)
    predicted_labels = (y_pred_test >= 0.5).float()
    accuracy = (predicted_labels == y_test).float().mean()
    print(f"\nTest Accuracy: {accuracy.item():.4f}")

# Plot Loss chart
plt.figure(figsize=(8, 3))
plt.plot(losses)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.title('Training Loss of Orbital Stability Prediction')
plt.grid(True)
plt.show()


