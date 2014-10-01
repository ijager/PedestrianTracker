import numpy as np
from iccv07 import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
from sklearn import svm
from sklearn.externals import joblib
from matplotlib.patches import Rectangle
from ParticleFilter import *

# === configuration === #

# select svm
dataset_name = "SVMs/newest/_svm_"
# select image set / video
im_set = iccv07(seq=3)
#number of particles
N_vector = range(10, 100, 25)
# number of iterations per N
iterations = 3

# number of weights to average over for the estimation
estimator_n = 10 
# initial particle distribution variance
var = 800;

# === global variables === #
automatic = False
bounds = []
evaluation = 0.0
svm = None
im_list = []
im_path = None
axis = None
particles = None
fig = None
hog = cv2.HOGDescriptor()
curr_im = None
p_filter = None
init_done = False

def getPatchFromParticle(particle, im):
	size = (64, 128)
	patch = im[int(particle[1]-size[1]/2):int(particle[1]+size[1]/2), int(particle[0]-size[0]/2):int(particle[0]+size[0]/2)]
	return patch

def computeLikelihood(patch):
	X = hog.compute(patch).T
	prediction = svm.predict_proba(X)
	return prediction[0][1]

def estimateLocation():
	global p_filter
	global estimator_n
	# find best matches
	ind = np.argsort(p_filter.weights, kind='mergesort')
	best = np.squeeze(p_filter.particles[ind][-estimator_n:])
	best_weights = np.squeeze(p_filter.weights[ind][-estimator_n:])
	# estimate location
	estimate = np.average(best, weights = best_weights, axis=0)
	confidence = np.average(best_weights, weights = best_weights, axis=0) 
	redrawRect(estimate, (np.maximum(1-confidence*2,0),np.minimum(confidence*2,1),0))
	return estimate

def evaluate(im_name, estimation, inset=(0,0,0,0)):
	global im_set
	annotation = im_set.get_coordinates(im_name) 
	ranges = [im_set.sort_coordinates(eval(c)) for c in annotation]
	if any(x_min+inset[0] <= estimation[0] <= x_max-inset[2] and y_min+inset[1] <= estimation[1] <= y_max-inset[3] for (x_min, y_min, x_max,y_max) in ranges):
		return True
	return False

def track():
	print 'start tracking'
	global im_path
	global im_list
	global p_filter
	global evaluation
	evaluation = 0
	for f in im_list[0:100]:	
		curr_im = cv2.imread(im_path + f,0)
		axis.set_data(curr_im)
		p_filter.resample()
		p_filter.moveGaussian(bounds=bounds)
		
		# compute liklihoods based on HOGDetector
		l = []
		
		for i in range(p_filter.num_particles):
			p = p_filter.particles[i,:]
			patch = getPatchFromParticle(p, curr_im)
			l.append(computeLikelihood(patch))
		p_filter.updateWeights(l)
		estimation = estimateLocation()
		evaluation += evaluate(f, estimation, inset = (0,0,0,0))
		redrawParticles()
		#pause the loop to give keyboard and mouse events a chance to get through
		plt.pause(0.02)
		plt.draw()
	return float(evaluation) / len(im_list)

def redrawRect(location, color="#ffffff", size = (64,128), centered=True):
	removeRect()
	drawRect(location, color, size, centered)

def removeRect():
	if len(plt.gca().patches) > 0:
		del plt.gca().patches[-1]

def drawRect(particle, color="#ffffff", size = (64,128), centered=True):
	if centered:
		r = Rectangle((particle[0]-size[0]/2, particle[1]-size[1]/2), size[0], size[1], facecolor="none", edgecolor=color)
	else:
		r = Rectangle((particle[0], particle[1]), size[0], size[1], facecolor="none", edgecolor=color)
	r.set_linestyle("dashed")
	r.set_linewidth(3)
	plt.gca().add_patch(r)

def redrawParticles():
	global particles
	particles.remove()
	drawParticles()

def drawParticles():
	global p_filter
	global particles
	particles = plt.scatter(p_filter.particles[:,0], p_filter.particles[:,1])

def initParticles(center, N):
	global p_filter
	global bounds
	global var
	print 'initialize', N, 'particles'
	p_filter = ParticleFilter(N, center, var, bounds=bounds)
	drawParticles()
	drawRect(center)
	plt.draw()


def onClick(event):
	global curr_im
	global init_done
	global var
	# nothing to do if no data exists
	if event.xdata == None or event.ydata == None:
		return
	if init_done != True:
		#print 'how many particles?'
		#n = int(raw_input('n = '))
		n = 40
		initParticles((event.xdata, event.ydata), n)
		init_done = True
	else:
		p_filter.relocateParticles((event.xdata, event.ydata), var, bounds=bounds)

def onKey(event):
	global init_done
	if event.key == "q" or event.key == "escape":
		print "exiting now"
		sys.exit(0)
	if event.key == " ":
		if init_done:
			plt.disconnect('key_release_event')
			track()

def autotest():
	global bounds
	global im_set
	results = []
	annotations = im_set.get_coordinates(im_set.get_image_list()[0])
	for N in N_vector:
		score = 0
		for iteration in range(iterations):
#			rand_ann = eval(np.random.choice(annotations))
			rand_ann = eval(annotations[0])
			start = ((rand_ann[0] + rand_ann[2])/2, (rand_ann[1] + rand_ann[3])/2)
			initParticles(start, N)
			score += track()
		results.append(score / iterations)
		removeRect()
		particles.remove()
	print results
	plotResults(results)

def plotResults(results):
	plt.figure()
	plt.plot(N_vector, results)
	plt.xlabel('Number of Particles')
	plt.ylabel('Particle Filter Score')
	plt.show()


if __name__ == "__main__":

	if len(sys.argv) > 1 and sys.argv[1] == "auto":
		print 'automatic testing mode'
		automatic = True
	try:
		svm = joblib.load("%s.pkl"%dataset_name)
	except:
		print "couldn't load svm:", dataset_name
		sys.exit(0)

	im_list = im_set.get_image_list()
	im_path = im_set.get_image_path()
	curr_im = cv2.imread(im_path + im_list[0],0)
	axis = plt.imshow(curr_im, cmap = cm.Greys_r)

	# fix axis
	size = curr_im.shape
	height = size[0]
	width = size[1]
	bounds = [32, 64, width-32, height-64]
	plt.ylim([height, 0])
	plt.xlim([0, width])

	if automatic:
		autotest()
	else:
		# connect mouse and keyboard events
		print 'Click on the image to select a location to initialize the particle filter, then start the tracking by pressing the space bar.'
		plt.connect('key_release_event', onKey)
		plt.connect('button_release_event', onClick)
		plt.show()

