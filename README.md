# De-pois-An-Attack-Agnostic-Defense-against-Data-Poisoning-Attacks
The most critical downside to AI is that its inefficiency is directly related to its data quality. Presently there are very few methods available that are used to protect the data from being attacked in the real time applications. But there is no commonplace in terms of defence technology that is being used for various types of attacks, in these situations the need for a generic method is highly needed for defending against those poisoning attacks, so we have come up with the defence technique that is called as De-Pois Defence method 
Python Packages
Dependencies:
Numpy 1.21.6
Pandas 1.3.5
Python 3.7.13
Pytorch 1.11.0+cu113
Keras 2.8.0
Scikit-learn 1.0.2
Scipy 1.4.1
art 5.6

Run the code and Applications
Run the code on the Google Colab . So you do not need to install any type of application .
Add all the .py files in the google drive along witH the MINST Dataset , then login Google Colab using the gmail account and access your saved files form the drive by providing the file path as an input .
Firstly execute Generator_CGAN_authen.py where in you have to change the accessed files path like MNIST dataset.
Then go ahead and run the Mimic_model_construction.py file similarly .
Lastly execute the Main.py file to get the respective results. Also execute Mnist_direct.py or Mnist_generative.py to generate the Poisoined samples required to run the code.

**Dataset:**
MNIST Datasbase - handwritten digits, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

Train and test datasets are found in the following link http://yann.lecun.com/exdb/mnist/

