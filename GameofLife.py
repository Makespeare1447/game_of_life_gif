
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import time
from IPython.display import clear_output, Image, display
import imageio
import scipy.misc
#import skimage.transform
import progressbar

from numba import jit


# In[2]:


def generate_map(sidelength):
    map = np.random.randint(2, size=(sidelength, sidelength))
    return map
    
def plot_map(map,fps):
    
    fig = plt.figure(figsize=(12,12))
    plt.imshow(map, cmap='binary')
    plt.show()
    plt.pause(1/fps)
    clear_output(wait=True)
    
@jit    
def count_nn(map, position):
    nn = 0
    x = position[0]
    y = position [1]
    if(map[x+1,y+1] == 1):
        nn = nn+1
    if(map[x-1,y+1] == 1):
        nn = nn+1
    if(map[x+1,y-1] == 1):
        nn = nn+1
    if(map[x-1,y-1] == 1):
        nn = nn+1
    if(map[x,y+1] == 1):
        nn = nn+1
    if(map[x,y-1] == 1):
        nn = nn+1
    if(map[x+1,y] == 1):
        nn = nn+1
    if(map[x-1,y] == 1):
        nn = nn+1
    else:
        pass
    return nn
        
@jit
def forward(map):
    rows = np.shape(map)[0]
    cols = np.shape(map)[1]
    newmap = np.zeros((rows,cols))
   
    for x in range(1,cols-1):
        for y in range(1,rows-1):
            nn = count_nn(map,(x,y))
            if(map[x,y]==0 and nn==3):
                newmap[x,y]=1
            if(map[x,y]==1 and nn<2):
                newmap[x,y]=0
            if(map[x,y]==1 and (nn==2 or nn==3)):
                newmap[x,y]=1
            if(map[x,y]==1 and nn>3):
                newmap[x,y]=0
                
    return newmap

@jit
def simplegameoflife(sidelength, iterations):
    map=generate_map(sidelength)
    for i in range(iterations):
        plot_map(map,100)
        map=forward(map)

@jit        
def scale_array(x, new_size):
    min_el = np.min(x)
    max_el = np.max(x)
    y = scipy.misc.imresize(x, new_size, mode='L', interp='nearest')
    y = y / 255 * (max_el - min_el) + min_el
    return y

@jit
def gameoflife_2_gif(sidelength, iterations):
    images = []
    map = generate_map(sidelength)
    bar = progressbar.ProgressBar(max_value=iterations)
    for i in range(iterations):
        scaledmap = scale_array(map,(1000,1000))
        images.append(scaledmap)
        map = forward(map)
        bar.update(i)
    imageio.mimsave('population.gif', images)


# In[6]:


gameoflife_2_gif(110,1000)

