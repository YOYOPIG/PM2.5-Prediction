# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# img = plt.imread("ncku.jpg")
# fig, ax = plt.subplots()
# ax.imshow(img)

# harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

# ax.imshow(harvest)

# # # We want to show all ticks...
# # ax.set_xticks(np.arange(len(farmers)))
# # ax.set_yticks(np.arange(len(vegetables)))
# # # ... and label them with the respective list entries
# # ax.set_xticklabels(farmers)
# # ax.set_yticklabels(vegetables)

# # # Rotate the tick labels and set their alignment.
# # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
# #          rotation_mode="anchor")

# # # Loop over data dimensions and create text annotations.
# # for i in range(len(vegetables)):
# #     for j in range(len(farmers)):
# #         text = ax.text(j, i, harvest[i, j],
# #                        ha="center", va="center", color="w")

# # ax.set_title("Harvest of local farmers (in tons/year)")
# # fig.tight_layout()
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from PIL import Image

# #2D Gaussian function
# def twoD_Gaussian(x, y, xo, yo, sigma_x, sigma_y):
#     a = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
#     c = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
#     g = np.exp( - (a*((x-xo)**2) + c*((y-yo)**2)))
#     return g.ravel()


# def transparent_cmap(cmap, N=255):
#     "Copy colormap and set alpha values"

#     mycmap = cmap
#     mycmap._init()
#     mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
#     return mycmap

# #Use base cmap to create transparent
# mycmap = transparent_cmap(plt.cm.Reds)

# # Import image and get x and y extents
# I = Image.open('./ncku.jpg')
# p = np.asarray(I).astype('float')
# w, h = I.size
# y, x = np.mgrid[0:h, 0:w]
# print(x.max())
# print(y.max())

# #Plot image and overlay colormap
# fig, ax = plt.subplots(1, 1)
# ax.imshow(I)
# Gauss = twoD_Gaussian(x, y, .9*x.max(), .4*y.max(), .6*x.max(), .6*y.max())
# cb = ax.contourf(x, y, Gauss.reshape(x.shape[0], y.shape[1]), 15, cmap=mycmap) # rendering data here
# plt.hold(True)
# plt.colorbar(cb) # CB on the side
# plt.show()



########################## 格子
import matplotlib
import matplotlib.image as mpimg 
from matplotlib.pyplot import show 
import numpy.random as random 
import seaborn as sns
import requests
import json
# get the map image as an array so we can plot it 

def get_pm25_avg(pos):
	r = requests.get(f'http://140.116.82.93:6800/campus/display/{pos}')
	data = json.loads(r.text)
	total_pm25 = 0
	for item in data:
		total_pm25 = total_pm25 + item.get('pm25')
	avg_pm25 = total_pm25/len(data)
	return avg_pm25

map_img = mpimg.imread('ncku.jpg') 



# making and plotting heatmap 
# heatmap_data = random.rand(50,50)
heatmap_data = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 5.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.7, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

heatmap_data[1][7] = get_pm25_avg(0)
heatmap_data[2][12] = get_pm25_avg(1)
heatmap_data[5][3] = get_pm25_avg(4)
heatmap_data[8][5] = get_pm25_avg(3)
heatmap_data[9][13] = get_pm25_avg(5)
heatmap_data[7][9] = get_pm25_avg(6)
heatmap_data[8][17] = get_pm25_avg(7)

# get row 8
delta = (heatmap_data[8][17] - heatmap_data[8][5])/12
heatmap_data[8][0] = heatmap_data[8][5] - 5 * delta
for i in range(19):
	heatmap_data[8][i+1] = heatmap_data[8][i] + delta

# get col 3
delta = (heatmap_data[8][3] - heatmap_data[5][3])/3
heatmap_data[0][3] = heatmap_data[5][3] - 5 * delta
for i in range(14):
	heatmap_data[i+1][3] = heatmap_data[i][3] + delta

# get col 7
delta = (heatmap_data[8][7] - heatmap_data[1][7])/7
heatmap_data[0][7] = heatmap_data[1][7] - 1 * delta
for i in range(14):
	heatmap_data[i+1][7] = heatmap_data[i][7] + delta

# get col 9
delta = (heatmap_data[8][9] - heatmap_data[7][9])
heatmap_data[0][9] = heatmap_data[7][9] - 7 * delta
for i in range(14):
	heatmap_data[i+1][9] = heatmap_data[i][9] + delta

# get col 12
delta = (heatmap_data[8][12] - heatmap_data[2][12])/6
heatmap_data[0][12] = heatmap_data[2][12] - 2 * delta
for i in range(14):
	heatmap_data[i+1][12] = heatmap_data[i][12] + delta

# get col 13
delta = (heatmap_data[9][13] - heatmap_data[8][13])/1
heatmap_data[0][13] = heatmap_data[8][13] - 8 * delta
for i in range(14):
	heatmap_data[i+1][13] = heatmap_data[i][13] + delta

# for j in range(20):
#     for i in range(15):
#         #print(heatmap_data[i][j])
#         print('yee')
# sns.set()

hmax = sns.heatmap(heatmap_data,
            cmap = matplotlib.cm.winter,
            alpha = 0.5, # whole heatmap is translucent
            annot = False,
            zorder = 2,
            )

# heatmap uses pcolormesh instead of imshow, so we can't pass through 
# extent as a kwarg, so we can't mmatch the heatmap to the map. Instead, 
# match the map to the heatmap:

hmax.imshow(map_img,
          aspect = hmax.get_aspect(),
          extent = hmax.get_xlim() + hmax.get_ylim(),
          zorder = 1) #put the map under the heatmap


show()