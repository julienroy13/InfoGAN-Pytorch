import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import os


"""
Texture generation using Perlin noise
"""
class NoiseUtils:
    
    def __init__(self, imageSize, angle, frequency, amplitude):
        self.imageSize = imageSize
        self.angle = angle
        self.frequency= frequency
        self.amplitude = amplitude
        self.gradientNumber = 256

        self.grid = [[]]
        self.gradients = []
        self.permutations = []
        self.img = np.zeros(shape=(self.imageSize, self.imageSize))

        self.__generateGradientVectors()
        self.__normalizeGradientVectors()
        self.__generatePermutationsTable()

    def __generateGradientVectors(self):
        for i in range(self.gradientNumber):
            while True:
                x, y = random.uniform(-1, 1), random.uniform(-1, 1)
                if x * x + y * y < 1:
                    self.gradients.append([x, y])
                    break

    def __normalizeGradientVectors(self):
        for i in range(self.gradientNumber):
            x, y = self.gradients[i][0], self.gradients[i][1]
            length = math.sqrt(x * x + y * y)
            self.gradients[i] = [x / length, y / length]

    # The modern version of the Fisher-Yates shuffle
    def __generatePermutationsTable(self):
        self.permutations = [i for i in range(self.gradientNumber)]
        for i in reversed(range(self.gradientNumber)):
            j = random.randint(0, i)
            self.permutations[i], self.permutations[j] = \
                self.permutations[j], self.permutations[i]

    def getGradientIndex(self, x, y):
        return self.permutations[(x + self.permutations[y % self.gradientNumber]) % self.gradientNumber]

    def perlinNoise(self, x, y):
        qx0 = int(math.floor(x))
        qx1 = qx0 + 1

        qy0 = int(math.floor(y))
        qy1 = qy0 + 1

        q00 = self.getGradientIndex(qx0, qy0)
        q01 = self.getGradientIndex(qx1, qy0)
        q10 = self.getGradientIndex(qx0, qy1)
        q11 = self.getGradientIndex(qx1, qy1)

        tx0 = x - math.floor(x)
        tx1 = tx0 - 1

        ty0 = y - math.floor(y)
        ty1 = ty0 - 1

        v00 = self.gradients[q00][0] * tx0 + self.gradients[q00][1] * ty0
        v01 = self.gradients[q01][0] * tx1 + self.gradients[q01][1] * ty0
        v10 = self.gradients[q10][0] * tx0 + self.gradients[q10][1] * ty1
        v11 = self.gradients[q11][0] * tx1 + self.gradients[q11][1] * ty1

        wx = tx0 * tx0 * (3 - 2 * tx0)
        v0 = v00 + wx * (v01 - v00)
        v1 = v10 + wx * (v11 - v10)

        wy = ty0 * ty0 * (3 - 2 * ty0)
        return (v0 + wy * (v1 - v0)) * 0.5 + 1
    
    def unit_vector(self, angle):
        """Creates a unit vector with given angle (in degrees)"""

        x = math.sin(angle * 2 * np.pi / 360)
        y = math.cos(angle * 2 * np.pi / 360)

        vector = np.array([x, y])

        return vector

    def makeTexture(self, texture=None):
        if texture is None:
            raise ValueError('You need to provide a texture')

        noise = np.zeros(shape=(self.imageSize, self.imageSize))
        max = min = None
        for i in range(self.imageSize):
            for j in range(self.imageSize):
                value = texture(i, j)
                noise[i, j] = value
                
                if max is None or max < value:
                    max = value

                if min is None or min > value:
                    min = value

        for i in range(self.imageSize):
            for j in range(self.imageSize):
                self.img[i, j] = (int) ((noise[i, j] - min) / (max - min) * 255 )

    def fractalBrownianMotion(self, x, y, func):
        octaves = 12
        current_ampl = self.amplitude
        frequency = 1.0 / self.imageSize
        persistence = 0.5
        value = 0.0
        for k in range(octaves):
            value += func(x * frequency, y * frequency) * current_ampl
            frequency *= 2
            current_ampl *= persistence
        return value
    
    def marble(self, x, y, noise = None):
        if noise is None:
            noise = self.perlinNoise
        
        real_freq = self.frequency / self.imageSize
        n = self.fractalBrownianMotion(8 * x, 8 * y, self.perlinNoise)
        
        u = np.expand_dims(np.array([x, y]), axis=0)
        v = np.expand_dims(self.unit_vector(self.angle), axis=1)
        r = np.dot(u, v)
        return (math.sin(r * real_freq + 4 * (n - 0.5)) + 1) * 0.5


if __name__ == "__main__":

	# OPTIONS
	IM_SIZE = 128
	N_IMAGES = int(50e3)
	SAVE_DIR = os.path.join("data", "synth")
	t_lim = [0, 360]
	f_lim = [10, 50]
	a_lim = [1., 3.]

	# =================================================
	if not os.path.exists(SAVE_DIR):
		os.makedirs(SAVE_DIR)

	# ANGLE VARIATIONS
	for i in tqdm(range(N_IMAGES)):
	    t = np.random.randint(low=t_lim[0], high=t_lim[1]+1)
	    f = np.random.randint(low=f_lim[0], high=f_lim[1]+1)
	    a = np.random.uniform(low=a_lim[0], high=a_lim[1])
	    a = float(int(100*a))/100.
	    
	    noise = NoiseUtils(imageSize=IM_SIZE, angle=t, frequency=f, amplitude=a)
	    noise.makeTexture(texture=noise.marble)

	    plt.imsave(os.path.join(SAVE_DIR, '{}_{}_{}.jpg'.format(t,f,int(100*a))), noise.img, cmap='gray')