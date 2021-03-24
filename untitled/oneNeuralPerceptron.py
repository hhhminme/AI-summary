def AND(x1,x2):
  w1,w2,t = 0.5,0.5,0.7
  sum = x1*w1 + x2*w2
  if(sum > t):
    print(1)
  elif sum <= t:
    print(0)

def OR(x1,x2):
  w1,w2,t = 1.0,1.0,0.9
  sum = x1*w1 + x2*w2
  if(sum > t):
    print(1)
  elif sum <= t:
    print(0)

def NAND(x1,x2):
  w1,w2,t = -0.5,-0.5,-0.7
  sum = x1*w1 + x2*w2
  if(sum > t):
    print(1)
  elif sum <= t:
    print(0)

def NOR(x1,x2):
  w1,w2,t = -1.0,-1.0,-0.9
  sum = x1*w1 + x2*w2
  if(sum > t):
    print(1)
  elif sum <= t:
    print(0)

def XOR(x1,x2):
  andw1,andw2,andt = 0.5,0.5,0.7
  notAndw1,notAndw2,notAndt = -0.5,-0.5,-0.7
  orw1,orw2,ort = 1.0,1.0,0.9

  notAndSum = x1*notAndw1 + x2*notAndw2
  orSum = x1*orw1 + x2*orw2

  if(notAndSum > notAndt):
    notAndSum = 1
  elif notAndSum <= notAndt:
    notAndSum = 0

  if(orSum > ort):
    orSum = 1
  elif orSum <= ort:
    orSum = 0

  andSum = andw1*notAndSum + andw2*orSum
  if(andSum > andt):
    print(1)
  elif andSum <= andt :
    print(0)

print("--AND--")
AND(0,0)
AND(1,0)
AND(0,1)
AND(1,1)
print("--OR--")
OR(0,0)
OR(1,0)
OR(0,1)
OR(1,1)
print("--NAND--")
NAND(0,0)
NAND(1,0)
NAND(0,1)
NAND(1,1)
print("--NOR--")
NOR(0,0)
NOR(1,0)
NOR(0,1)
NOR(1,1)
print("--XOR--")
XOR(0,0)
XOR(1,0)
XOR(0,1)
XOR(1,1)