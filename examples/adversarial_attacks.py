import matplotlib.pyplot as plt
import numpy as np

from extra.datasets import fetch_mnist
from tinygrad.nn import Linear, Conv2d
from tinygrad.nn.optim import SGD
from tinygrad.state import get_parameters
from tinygrad.tensor import Tensor

###########
## MODEL ##
###########

class TinyCNN:
    def __init__(self):
        self.cnn1 = Conv2d(1, 32, 3)
        self.cnn2 = Conv2d(32, 32, 7)
        self.l1 = Linear(128, 64, bias=True)
        self.l2 = Linear(64, 10, bias=False)

    def __call__(self, x):
        x = self.cnn1(x).relu().max_pool2d(kernel_size=(3,3))
        x = self.cnn2(x).relu()
        x = x.reshape(x.shape[0], -1)
        x = x.relu()
        x = self.l1(x)
        x = x.relu()
        logits = self.l2(x)
        return logits

####################
# TRAINING HELPERS #
####################

def train_model(
        network, 
        loss_fn, 
        optimizer,
        X_train, Y_train, 
        X_val, Y_val, 
        n_batch, n_epochs
    ):
    Tensor.train = True
    X_val        = Tensor(X_val, requires_grad = False)

    for ep in range(n_epochs):
        # Retrieves data
        indexes = np.random.randint(0, X_train.shape[0], size=(n_batch,))
        data    = Tensor(X_train[indexes], requires_grad = False)
        labels  = Y_train[indexes]
        # Sets gradients to 0
        optimizer.zero_grad()
        # Forward pass
        outputs = network(data)
        loss    = loss_fn(outputs, labels)
        # Backward propagation
        loss.backward()
        optimizer.step()
        # Compute accuracy
        preds    = np.argmax(outputs.numpy(), axis=-1)
        accuracy = np.sum(preds == labels)/len(labels)
        if ep % 50 == 0:
            Tensor.training = False
            out     = network(X_val)
            preds   = np.argmax(out.softmax().numpy(), axis=-1)
            val_acc = np.sum(preds == Y_val)/len(Y_val)
            Tensor.training = True
            print(f"\tEpoch {ep} | " \
                  f"Loss: {loss.numpy():.2f} | " \
                  f"Train acc: {100 * accuracy:.2f}% | "\
                  f"Val. acc: {100 * val_acc:.2f}%")
    
    return network
    
def one_hot_encoding(labels: np.array, n_classes: int):
    """One-hot encodes a set of integer labels (e.g. [0, ..., 9])
    """
    flat_labels       = labels.flatten().astype(np.int32)
    sparsified_labels = np.zeros((labels.shape[0], n_classes), np.float32)
    sparsified_labels[range(sparsified_labels.shape[0]), flat_labels] = -1.
    target_shape      = list(labels.shape) + [n_classes]
    sparsified_labels = sparsified_labels.reshape(target_shape)
    return Tensor(sparsified_labels)

def sparse_cross_entropy(logits: Tensor, labels: np.array, reduction: str = "sum"):
    """Implements the cross entropy loss 
    """
    n_classes = logits.shape[-1]
    ohe       = one_hot_encoding(labels, n_classes)
    if reduction == "sum":
        return logits.log_softmax().mul(ohe).sum()
    elif reduction == "mean":
        return logits.log_softmax().mul(ohe).mean()
    else:
        return logits.log_softmax().mul(ohe)

##############################
# ADVERSARIAL ATTACK HELPERS #
##############################

def tensor_norm(x: Tensor, p: float):
    """Computes the p-norm of a tensor.
    """
    assert p in [2., float("inf")]
    shape = x.shape
    if p == 2:
        norm = x.pow(2).reshape(x.shape[0], -1).sum(axis=1).sqrt().squeeze()
    elif p == float("inf"):
        norm = x.reshape(x.shape[0], -1).max(axis=1)
    norm = norm.reshape(norm.shape[0], *[1]*len(shape[1:]))
    return norm

class Adversarial_Buffer:

    def __init__(
            self, 
            features,
            random_initialization: bool = False,
            norm: float = None,
            epsilon: float = None
        ):
        """Initialization method for adversarial perturbation buffer.
        """
        # Declares the placeholder perturbation
        if random_initialization:
            self.perturbation = getattr(Tensor, "rand")(*features).sub(1/2).mul(2)
        else:
            self.perturbation = getattr(Tensor, "zeros")(*features)
        
        # If indicated, initializes the placeholder perturbation
        if random_initialization and \
           norm is not None and \
           epsilon is not None and epsilon > 0:
            normalizing_const = tensor_norm(self.perturbation, norm)
            self.perturbation = self.perturbation.div(normalizing_const).mul(epsilon)

        self.perturbation.requires_grad = True

    def __call__(self, x):
        """Method that allows modularization
        """
        x = x.add(self.perturbation)
        return x
    
    def __str__(self):
        return str(self.perturbation.numpy())
    
class Adversarial_Attack():

    def __init__(self: object,
                 loss_function: object,
                 n_iterations: int = 1,
                 random_initialization: bool = False,
                 norm: float = None,
                 epsilon: float = None):
        """Initialization method for Adversarial Attack class.
        """
        
        # Basic checks
        assert type(n_iterations)==int and n_iterations > 0 #<n_iterations> must be positive int
        assert type(random_initialization)==bool # <random_initialization> must be a bool
        assert norm in [2., float("inf")]
        assert epsilon is None or type(epsilon)==float and epsilon > 0 # <epsilon> is a positive, real-valued normalizing constant
        
        # Prints the attack type
        if n_iterations == 1:
            if random_initialization:
                print("\tR+FGSM Adversarial attack declared.")
            else:
                print("\tFGSM Adversarial attack declared.")
        else:
            print("\tProjected Gradient Descent (PGD) Adversarial attack declared.")

        self.loss = loss_function
        self.iter = n_iterations
        self.rand = random_initialization
        self.norm = norm
        self.cons = epsilon

    def run(self: object,
            X: Tensor, 
            Y: Tensor,
            network: object,
            target_class: int = 0):
        """Runs the declared adversarial attack.
        """
        # Sets flags
        Tensor.train = True
        network.training = False

        # Declares array of targets (needs to be numpy for loss func)
        targets = Tensor.full(Y.shape, fill_value=target_class).numpy()

        # Declares adversarial buffer
        perturbations = Adversarial_Buffer(X.shape, self.rand, self.norm, self.cons)

        if self.iter == 1:
            # Retrieves loss
            out = network(perturbations(X))
            loss = self.loss(out, targets).mul(-1.)
            loss.backward()
            # Compute perturbation
            perturbation = perturbations.perturbation
            perturbation = perturbation.grad.sign().mul(self.cons)
            # Clamps perturbation
            perturbation = X.add(perturbation).maximum(0).minimum(1).sub(X)

            return perturbation

        # Iterates the attack
        for _ in range(self.iter):
            # Retrieves loss
            out = network(perturbations(X))
            loss = self.loss(out, targets).mul(-1.)
            loss.backward()
            # Compute perturbation update
            perturbation = perturbations.perturbation
            update = perturbation.grad
            norm = tensor_norm(update, self.norm)
            update = update.div(norm).mul(self.cons/self.iter if self.norm!=2 else self.cons*0.2)
            perturbation = perturbation.add(update).realize()
            # Clamps perturbation
            perturbation = X.add(perturbation).maximum(0).minimum(1).sub(X)
            # Feed update back into buffer
            perturbations.perturbation = perturbation.realize()
        
        return perturbations.perturbation
            
def attack_network(
        network: object, 
        loss_fn: object,
        X_test: np.array, 
        Y_test: np.array,
        n_iterations: int = 1,
        random_initialization: bool = False,
        norm: float = 2,
        epsilon: float = 0.3,
        target_class: int = 3,
        print_example: list = None
    ):
    """Function to test an attack on a network using first 128 elements of given set
    """

    X_test = Tensor(X_test[:128])
    Y_test = Y_test[:128]

    out      = network(X_test)
    preds    = np.argmax(out.softmax().numpy(), axis=-1)
    test_acc = np.sum(preds == Y_test)/len(Y_test)

    print(f"\tClean test accuracy: {100 * test_acc:.2f}%")

    attack = Adversarial_Attack(loss_fn, n_iterations, random_initialization, norm, epsilon)
    perturbations = attack.run(X_test, Y_test, network, target_class)

    targets = Tensor.full(Y_test.shape, fill_value=target_class).numpy()
    inputs  = X_test.add(perturbations).minimum(1).maximum(0)
    outputs = network(inputs)
    preds   = np.argmax(outputs.numpy(), axis=-1)
    # print(preds, targets)
    adv_acc = np.sum(preds == Y_test)/len(Y_test)
    asr     = np.sum(preds == targets)/len(Y_test)

    print(f"\tAdversarial test accuracy: {100 * adv_acc:.2f}% | ASR: {100 * asr:.2f}%")

    if print_example is not None and print_example[0]:
        plt.imshow(inputs.numpy()[2].reshape(28, 28), cmap="gray")
        plt.savefig(print_example[1])

    return None
    
if __name__ == "__main__":

    # Sets variables
    n_epochs = 1000
    n_valid  = 10000

    # Retrieve dataset
    X_train, Y_train, X_test, Y_test = fetch_mnist()
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    # Normalize
    X_train /= 255.
    X_test  /= 255.

    # Split training and validation sets
    indexes_valid = np.random.choice(range(len(X_train)), 
                                     n_valid, 
                                     replace=False)
    mask = np.ones(Y_train.shape, bool)
    mask[indexes_valid] = False

    X_val, Y_val = X_train[indexes_valid], Y_train[indexes_valid]
    X_train, Y_train = X_train[mask], Y_train[mask]

    # Generate network
    network    = TinyCNN()
    net_params = get_parameters(network)
    optimizer  = SGD(net_params, lr=0.0005)

    # Train model
    print("\nTrains a 4-layer CNN (2-conv2d, 2 linear) network (for attack purposes)")
    network = train_model(
        network, 
        sparse_cross_entropy, 
        optimizer,
        X_train, 
        Y_train, 
        X_val, 
        Y_val,
        32,
        n_epochs
    )

    # Runs FGSM attack
    print("\nRuns a FGSM attack with 2-norm, 0.3-eps, trg=3")
    perturbation = attack_network(
        network, 
        sparse_cross_entropy, 
        X_test, 
        Y_test,
        n_iterations = 1,
        random_initialization = False,
        norm = 2,
        epsilon = 0.3,
        target_class = 3,
        print_example = [True, "FGSM.png"]
    )

    # Runs R+FGSM attack
    print("\nRuns a R+FGSM attack with 2-norm, 0.3-eps, trg=3")
    perturbation = attack_network(
        network, 
        sparse_cross_entropy, 
        X_test, 
        Y_test,
        n_iterations = 1,
        random_initialization = True,
        norm = 2,
        epsilon = 0.3,
        target_class = 3,
        print_example = [True, "R_FGSM.png"]
    )

    # Runs PGD attack
    print("\nRuns a PGD attack with 40-iter, 2-norm, 0.3-eps, trg=3")
    perturbation = attack_network(
        network, 
        sparse_cross_entropy, 
        X_test, 
        Y_test,
        n_iterations = 40,
        random_initialization = True,
        norm = 2,
        epsilon = 0.3,
        target_class = 3,
        print_example = [True, "PGD_l2.png"]
    )

    # Runs test norm=inf attack
    print("\nRuns a PGD attack with 40-iter, inf-norm, 0.3-eps, trg=3")
    perturbation = attack_network(
        network, 
        sparse_cross_entropy, 
        X_test, 
        Y_test,
        n_iterations = 40,
        random_initialization = True,
        norm = float("inf"),
        epsilon = 0.3,
        target_class = 3,
        print_example = [True, "PGD_linf.png"]
    )
