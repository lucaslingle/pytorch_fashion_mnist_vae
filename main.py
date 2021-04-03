import torch as tc
from torch.nn import functional as F
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np

training_data = tv.datasets.FashionMNIST(
    root='data', train=True, download=True, transform=tv.transforms.ToTensor())

test_data = tv.datasets.FashionMNIST(
    root='data', train=False, download=True, transform=tv.transforms.ToTensor())

# Create data loaders.
batch_size = 64
train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


class RecognitionModel(tc.nn.Module):
    def __init__(self, z_dim):
        super(RecognitionModel, self).__init__()
        self.flattener = tc.nn.Flatten()
        self.enco_stack = tc.nn.Sequential(
            tc.nn.Linear(28*28, 400),
            tc.nn.ReLU(),
            tc.nn.Linear(400, z_dim * 2)
        )

    def forward(self, x):
        flat = self.flattener(x)
        code_pre = self.enco_stack(flat)
        mu_z, logvar_z = tc.chunk(code_pre, 2, dim=-1)
        return mu_z, logvar_z


class Generator(tc.nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.deco_stack = tc.nn.Sequential(
            tc.nn.Linear(z_dim, 400),
            tc.nn.ReLU(),
            tc.nn.Linear(400, 28 * 28 * 1)
        )

    def forward(self, x):
        logits_flat = self.deco_stack(x)
        logits_square = tc.reshape(logits_flat, (-1, 1, 28, 28))
        return logits_square


class VAE(tc.nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.recognition_model = RecognitionModel(z_dim)
        self.generator = Generator(z_dim)

    def forward(self, x):
        mu_z, logvar_z = self.recognition_model(x)
        z = mu_z + tc.exp(0.5 * logvar_z) * tc.randn_like(mu_z)
        px_given_z_logits = self.generator(z)
        px_given_z_probs = tc.nn.Sigmoid()(px_given_z_logits)
        return px_given_z_probs, mu_z, logvar_z

    def decode(self, z):
        px_given_z_logits = self.generator(z)
        px_given_z_probs = tc.nn.Sigmoid()(px_given_z_logits)
        return px_given_z_probs

    def sample(self, num_samples):
        z = tc.randn(size=(num_samples, self.z_dim))
        px_given_z_probs = self.decode(z)
        return px_given_z_probs



def kl(mu_z, logvar_z):
    """
    Compute the KL divergence for a batch of data.
    :param mu_z:
    :param logvar_z:
    :return:
    """
    kl_term_1 = logvar_z.exp()  # sigma^2.
    kl_term_2 = mu_z.pow(2)  # mu^2
    kl_term_3 = -1  # -k
    kl_term_4 = -logvar_z  # -log(sigma^2)

    batch_size = len(mu_z)
    kl_div = 0.5 * (kl_term_1 + kl_term_2 + kl_term_3 + kl_term_4).sum() / batch_size
    return kl_div

def reconstruction_loss(x, x_recon):
    """
    Compute the KL divergence for a batch of data.
    :param x: batch of binarized images.
    :param x_recon: batch of probabilities of a white pixel ('1') for binarized image data.
    :return: negative log probability of reconstruction of the input.
    """
    pxz = x_recon
    batch_size = len(x)
    neg_log_px_given_z = F.binary_cross_entropy(input=pxz, target=x, reduction='sum') / batch_size
    return neg_log_px_given_z

def elbo(x, x_recon, mu_z, logvar_z):
    """
    Compute the ELBO for a batch of data.
    :param x:
    :param x_recon:
    :param mu_z:
    :param logvar_z:
    :return:
    """
    log_px_given_z = -reconstruction_loss(x, x_recon)
    dkl_z = kl(mu_z, logvar_z)
    elbo_x = log_px_given_z - dkl_z
    return elbo_x

def loss_fn(x_recon, mu_z, logvar_z, x):
    return -elbo(x, x_recon, mu_z, logvar_z)

device = "cuda" if tc.cuda.is_available() else "cpu"
model = VAE(z_dim=100).to(device)
print(model)

optimizer = tc.optim.Adam(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    num_training_examples = len(dataloader.dataset)

    for batch, (X, _) in enumerate(dataloader):
        X = X.to(device)

        # Compute prediction error
        x_recon, mu_z, logvar_z = model(X)
        loss = loss_fn(x_recon, mu_z, logvar_z, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            current_idx = batch_size * (batch-1) + len(X)
            print(f"loss: {loss.item():>7f}  [{current_idx:>5d}/{num_training_examples:>5d}]")


def test(dataloader, model):
    num_test_examples = len(dataloader.dataset)
    model.eval()
    test_loss = 0.0
    with tc.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            x_recon, mu_z, logvar_z = model(X)
            test_loss += len(X) * loss_fn(x_recon, mu_z, logvar_z, X).item()
    test_loss /= num_test_examples
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)

print("Done!")

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    input_example = X[0]
    output_example = model.forward(tc.unsqueeze(input_example, dim=0))[0].detach()[0]
    img = np.transpose(np.concatenate([input_example, output_example], axis=-1), axes=[1,2,0])
    img_3channel = np.concatenate([img for _ in range(0,3)], axis=-1)
    plt.imshow(img_3channel)
    plt.show()
    break

output_examples = model.sample(num_samples=8).detach()
output_examples = np.transpose(output_examples, axes=[0, 2, 3, 1])
img = np.concatenate([output_examples[i] for i in range(0,8)], axis=1)
img_3channel = np.concatenate([img for _ in range(0, 3)], axis=-1)
plt.imshow(img_3channel)
plt.show()