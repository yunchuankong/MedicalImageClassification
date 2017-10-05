import numpy as np

file = np.load("/home/ykong24/stroke_processed_data.npz")
image_data = file['x']
# label_data = file['y']

image_data_resized = []
for i, instance in enumerate(image_data):
	pile = []
	for j, slice in enumerate(instance):
		 temp = slice.reshape((128, 128))
		 temp = temp[::2,::2]
		 temp = temp.reshape((64,64,1))
		 pile.append(temp)
	pile = np.array(pile)
	image_data_resized.append(pile)
	print np.shape(pile)
	print ("Pile attached: %d" %(i))

image_data_resized = np.array(image_data_resized)

np.savez('/home/ykong24/stroke_resized_data.npz', x=image_data_resized, y=file['y'])
		 
