"""Author: Nagabhushan S Baddi\nCentral Paramater Server Process"""

from socket import *
import numpy as np

#Server Process

class ServerProcess:
    """Class representing the central parameter server"""

    port = 1234                   # port number
    numberOfOrganizations = 5	  # Number of Client Organizations training collaboratively

    def __init__(self, layers, nb_epochs):
    	#Organizations should decide on the architecture of the neural network i.e., neurons in the hidden layer
    	#Server initializes its list of parameters accordingly

    	self.layers = layers  #layers is the list of the number of neurons in the hidden layers
        self.numLayers = len(layers)
        self.nb_epochs = nb_epochs
        

        #Initialize the parameters (biases and weights)
        self.biases = [np.random.uniform(-4 * sqrt(6 / (layers[x - 1] + layers[x])), 4 * sqrt(6 / (layers[x - 1] + layers[x])),(layers[x], 1)) for x in range(1, self.numLayers)]
        self.weights = [np.random.uniform(-4 * sqrt(6 / (layers[x - 1] + layers[x])), 4 * sqrt(6 / (layers[x - 1] + layers[x])),(layers[x], layers[x - 1])) for x in range(1, self.numLayers)]
       

      	#Initialize Stat Vector
        self.statForBiases = [np.zeros((self.layers[x], 1)) for x in range(1, self.numLayers)]
        self.statForWeights = [np.zeros((self.layers[x], self.layers[x - 1])) for x in range(1, self.numLayers)]


        #Establish Network Connection
        sock = socket(AF_INET, SOCK_STREAM) 
	sock.bind(('', self.port)) 
	sock.listen(self.numberOfOrganizations) 

	
	#execute and wait for events
	execute()


    
    def execute(self):
    	"""This method waits for events (requests) from the clients related
    	 to upload and download and processes them"""

	while True:
		conn, addr = sock.accept()			
		typeOfEvent = conn.recv(10) 
			
		#if typeOfEvent is download event
		handleDownload(conn)

		#else if upload event
		handleUpload(conn)



    def handleDownload(self, conn):
	"""This method handles download of a fraction of the parameters according to the devised algorith"""

	#Handle Download Requests according to Figure 2.



    def handleUpload(self, conn):
	"""This method handles upload of the gradients vectors according to the devised algorith"""

	#Handle Upload Requests according to Figure 2.



if __name__ == "__main__":
    server = ServerProcess([100, 30], nb_epochs=30)
