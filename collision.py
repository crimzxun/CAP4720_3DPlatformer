import pyrr
import random

def check_collision(obj, plane, trans, scale):
    if(obj[3][1] - 0.1 <= plane[3][1]):
        obj[3][1] = plane[3][1] - 0.008
    elif(obj[3][1] > plane[3][1]):
        obj = pyrr.matrix44.multiply(trans, scale)

def check_collision4(obj,gift,count):
    if(((obj[3][0] >= gift[3][0] - 0.20) and (obj[3][0] <= gift [3][0] + 0.18)) and ((obj[3][2] >= gift[3][2] - 0.36) and (obj[3][2] <= gift[3][2] + 0.58))):
        print("you picked up a gift")
        gift[3][0] = random.uniform(-1.6,1.6)
        gift[3][2] = random.uniform(-1.6,1.6)
        count = count + 1
        print("Gifts: ", count)
        return count
    return count
        

def check_collison3(obj,tree,ty,tx):
    if(obj[3][0] >= tree[3][0] - 0.66):
        print("collision with tree")
        ty = 0
# and (obj[3][0] <= tree[3][0] + 0.66)) and ((obj[3][2] >= tree[3][2] - 0.75) and (obj[3][2] <= tree[3][2] +1))):
        