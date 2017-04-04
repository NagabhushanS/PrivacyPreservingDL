"""Author: Nagabhushan S Baddi\nOrganization Client Process"""

#Client Process

class ClientProcess:
	"""Class representing an organization neural network client"""

	port = 1234
	host = 'examplehost.com'
	local_dataset = None

	def __init__(self, layers, nb_epochs, local_dataset):
		#Organizations should decide on the architecture of the neural network i.e., neurons in the hidden layer
    		#Each Client initializes its list of parameters accordingly

    		self.layers = layers  #layers is the list of the number of neurons in the hidden layers
        	self.numLayers = len(layers)
        	self.nb_epochs = nb_epochs
        	self.local_dataset = local_dataset


        	#Initialize the parameters (biases and weights)
        	self.biases = [np.random.uniform(-4 * sqrt(6 / (layers[x - 1] + layers[x])), 4 * sqrt(6 / (layers[x - 1] + layers[x])),(layers[x], 1)) for x in range(1, self.numLayers)]
        	self.weights = [np.random.uniform(-4 * sqrt(6 / (layers[x - 1] + layers[x])), 4 * sqrt(6 / (layers[x - 1] + layers[x])),(layers[x], layers[x - 1])) for x in range(1, self.numLayers)]
       

      		#Initialize Stat Vector
        	self.statForBiases = [np.zeros((self.layers[x], 1)) for x in range(1, self.numLayers)]
        	self.statForWeights = [np.zeros((self.layers[x], self.layers[x - 1])) for x in range(1, self.numLayers)]


        	#Connect to the server Process
        	sock = socket(AF_INET, SOCK_STREAM)
		sock.connect((self.host, self.port)) 
		

		#Run SGD on local dataset and download and upload to serve
		execute(sock)


	def execute(self, sock):
		"""This method is the main method of the client process. It runs SGD on the local dataset and handles
		communications with the server process"""

		for epoch in range(self.epochs):
            
           		#download a fraction of parameters from the server
			self.downloadFromServer(sock)

			#run SGD on the local dataset
			self.SGD()

			#upload a fraction of the gradients of the parameters to the server
			self.uploadToServer(sock)

			
	def SGD(self):
		"""This method runs the stochastic gradient descent on the local dataset and updates the parameters
		of the local neural network"""


		#runs sgd on the local dataset and update the parameters


	def propBack(self):
		"""This method implements the back-propagation algorithm used to compute the error vectors of the 
		hidden layers. This method is used internally by the method SGD()"""

		#computes error vectors of the hidden layers required in the method SGD().


	def uploadToServer(self, conn):
		"""This method uploads a fraction of gradients of the parameters to the server"""

		#selects the set of parameters whose gradients are to be uploaded according to the algorithm in 
		#figure 1 and then uploads these to the central server.
		conn.send(gradient_bytes)


	def downloadFromServer(self, conn):
		"""This method downloads a fraction of parameters from the server"""

		#Downloads a fraction of the parameters from the central server.
		downloaded_parameters = conn.send(parameter_bytes)

		#then replaces the corresponding parameters with these downloaded parameters

		
def load_dataset():
	"""Method to load the local-dataset from the disk"""
	
	#loads the local dataset
	#returns local dataset
	

if __name__ == '__main__':
	#load local dataset
	dataset = load_dataset()

	#create a client process
	client  = ClientProcess([100, 30], nb_epochs=30, local_dataset=dataset)
