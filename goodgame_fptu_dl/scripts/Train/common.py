import json

def getConfig(name):
    with open('config.json') as f:
        data = json.load(f)
    
    return data[name]

def getConfigTest(name):
    with open('config.json') as f:
        data = json.load(f)
    
    return data[name]

# TINH D_MAX KHI NORMALIZE DISTANCE
def calculateNormalizeDistances():
    image = cv.imread('normalize.png')
    h = image.shape[0]
    w = image.shape[1]

    view_point = np.array([w//2, h], dtype=np.int)

    points = np.zeros((5, 2), dtype=np.int)
    layer_height = 20

    for i in range(5):
        r = 10 + (11 + i) * layer_height
        points[i][1] = r
        points[i][0] = ((image[r][:,0] != 0) | (image[r][:,1] != 0) | (image[r][:,2] != 0)).argmax(axis=0)
        
    distances = np.sqrt(np.sum((points - view_point)**2, axis=1))
    return distances[::-1]