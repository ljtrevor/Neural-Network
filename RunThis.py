import mnist_loader
import network

class RunThis():
	def main():
	
		training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

		net = network.Network([784,10])

		net.SGD(training_data, 30, 10, 3.0, test_data = test_data)
		
		#SGD(self, training_data, epochs, mini_batch_size, eta,
        #test_data=None):
	

	if __name__ == "__main__": main()