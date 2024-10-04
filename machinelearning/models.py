import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        pred = self.run(x)
        if nn.as_scalar(pred) >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        while True:
            converged = True
            for x, y in dataset.iterate_once(1):
                pred = self.get_prediction(x)
                corr = nn.as_scalar(y)
                if pred == corr:
                    continue
                self.w.update(x, corr)  
                converged = False
            if converged:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        hidden_size = 512
        self.w_1 = nn.Parameter(1, hidden_size)
        self.b_1 = nn.Parameter(1, hidden_size)
        self.w_2 = nn.Parameter(hidden_size, 1)
        self.b_2 = nn.Parameter(1, 1)
        self.transforms = [self.w_1, self.b_1, self.w_2, self.b_2]


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        #want pred = relu(0, (xw_1 + b_1)) * w_2 + b_2
        xw_1 = nn.Linear(x, self.w_1)
        relu_term = nn.ReLU(nn.AddBias(xw_1, self.b_1))
        x_term = nn.Linear(relu_term, self.w_2)
        pred = nn.AddBias(x_term, self.b_2)
        return pred


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        pred = self.run(x)
        return nn.SquareLoss(pred, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        learning_rate = 0.05
        batch_size = 200
        stoploss = float('inf')
        while stoploss > 0.02:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, self.transforms)
                for i in range(len(self.transforms)):
                    self.transforms[i].update(grads[i], -learning_rate)
                stoploss = nn.as_scalar(self.get_loss(x, y))





class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        hidden_size = 200
        self.w_1 = nn.Parameter(784, hidden_size)
        self.b_1 = nn.Parameter(1, hidden_size)
        self.w_2 = nn.Parameter(hidden_size, 10)
        self.b_2 = nn.Parameter(1, 10)
        self.transforms = [self.w_1, self.b_1, self.w_2, self.b_2]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        xw_1 = nn.Linear(x, self.w_1)
        relu_term = nn.ReLU(nn.AddBias(xw_1, self.b_1))
        x_term = nn.Linear(relu_term, self.w_2)
        pred = nn.AddBias(x_term, self.b_2)
        return pred

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        preds = self.run(x)
        return nn.SoftmaxLoss(preds, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        learning_rate = 0.5
        batch_size = 100
        val_acc = 0
        while val_acc < 0.975:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, self.transforms)
                for i in range(len(self.transforms)):
                    self.transforms[i].update(grads[i], -learning_rate)
            val_acc = dataset.get_validation_accuracy()


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        hidden_size = 150
        self.w_1 = nn.Parameter(self.num_chars, hidden_size)
        self.b_1 = nn.Parameter(1, hidden_size)
        self.w_2 = nn.Parameter(hidden_size, hidden_size)
        self.b_2 = nn.Parameter(1, hidden_size)
        self.w_1_hidden = nn.Parameter(hidden_size, hidden_size)
        self.b_1_hidden = nn.Parameter(1, hidden_size)
        self.w_2_hidden = nn.Parameter(hidden_size, hidden_size)
        self.b_2_hidden = nn.Parameter(1, hidden_size)
        self.w_final = nn.Parameter(hidden_size, 5)
        self.b_final = nn.Parameter(1,5)
        self.transforms = [self.w_1, self.b_1, self.w_2, self.b_2, self.w_1_hidden, self.b_1_hidden, self.w_2_hidden, self.b_2_hidden, self.w_final, self.b_final]


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        for i, char in enumerate(xs):
            if i==0:
                nonlinear_z_0 = nn.ReLU(nn.AddBias(nn.Linear(char, self.w_1), self.b_1))
                hidden = nn.AddBias(nn.Linear(nonlinear_z_0, self.w_2), self.b_2)
            else:
                nonlinear_combined_z1 = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(char, self.w_1), nn.Linear(hidden, self.w_1_hidden)), self.b_1_hidden))
                hidden = nn.ReLU(nn.AddBias(nn.Linear(nonlinear_combined_z1, self.w_2_hidden), self.b_2_hidden))
        return nn.AddBias(nn.Linear(hidden, self.w_final), self.b_final)


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        preds = self.run(xs)
        return nn.SoftmaxLoss(preds, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        learning_rate = 0.2
        batch_size = 100
        val_acc = 0
        while val_acc < 0.85:
            for x,y in dataset.iterate_once(batch_size): 
                grads = nn.gradients(self.get_loss(x,y), self.transforms)
                for i in range(len(self.transforms)):
                    self.transforms[i].update(grads[i], -learning_rate)
            val_acc = dataset.get_validation_accuracy()
