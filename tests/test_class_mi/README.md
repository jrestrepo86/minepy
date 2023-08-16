# Minepy

Mutual information neural estimation classification based module

## Classifier based  Mutual Information Estimation

```py
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from minepy.class_mi.class_mi import ClassMI

# Net
input_dim = 2
model_params = {"hidden_dim": 50, "afn": "relu", "num_hidden_layers": 3}

mu = np.array([0, 0])
Rho = np.linspace(-0.98, 0.98, 21)
mi_teo = np.zeros(*Rho.shape)
class_mi = np.zeros(*mi_teo.shape)

# Training
batch_size = 300
max_epochs = 3000
train_params = {
    "batch_size": batch_size,
    "max_epochs": max_epochs,
    "lr": 1e-3,
    "lr_factor": 0.1,
    "lr_patience": 10,
    "stop_patience": 100,
    "stop_min_delta": 0.01,
    "verbose": False,
}
for i, rho in enumerate(tqdm(Rho)):
    # Generate data
    cov_matrix = np.array([[1, rho], [rho, 1]])
    joint_samples_train = np.random.multivariate_normal(
        mean=mu, cov=cov_matrix, size=(10000, 1)
    )
    X = np.squeeze(joint_samples_train[:, :, 0])
    Z = np.squeeze(joint_samples_train[:, :, 1])

    # Teoric value
    mi_teo[i] = -0.5 * np.log(1 - rho**2)
    # models
    class_mi_model = ClassMI(input_dim, **model_params)
    # Train models
    class_mi_model.fit(X, Z, **train_params)
    # Get mi estimation
    class_mi[i] = class_mi_model.get_mi()

# Plot
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
ax.plot(Rho, mi_teo, ".k", label="True mi")
ax.plot(Rho, class_mi, "b", label="Class mi")
ax.legend(loc="upper center")
ax.set_xlabel("rho")
ax.set_ylabel("mi")
ax.set_title("Classification based mutual information")
plt.show()
```

### Classifier based Conditional Mutual Information Estimation

```py
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.class_mi.class_cmi_gen import ClassCMIGen
from minepy.minepy_tools import coupledHenon, embedding


def cmi(target, source, u, emb_params, model_params, train_params):
    target = embedding(target, **emb_params)
    source = embedding(source, **emb_params)

    n = target.shape[0]
    target_u = target[u:, :]
    target = target[: n - u, :]
    source = source[: n - u, :]
    # I(target_u, source, target) - I(target_u,target)
    class_cmi_model = ClassCMIGen(target_u, source, target, **model_params)
    # training
    class_cmi_model.fit(**train_params)
    # Get mi estimation
    cmi_test, cmi_val = class_cmi_model.get_cmi()
    # Get curves
    (
        Dkl_val,
        val_loss,
        val_acc,
    ) = class_cmi_model.get_curves()
    return {
        "cmi_test": cmi_test,
        "cmi_val": cmi_val,
        "Dkl_val_epoch": Dkl_val,
        "val_loss_epoch": val_loss,
        "val_acc_epoch": val_acc,
    }

emb_params = {"m": 3, "tau": 2}
u = 1
# model
model_params = {
    "hidden_dim": 128,
    "num_hidden_layers": 3,
    "afn": "relu",
}
# embedding parameters
batch_size = 64
max_epochs = 8000
train_params = {
    "batch_size": batch_size,
    "max_epochs": max_epochs,
    "knn": 10,
    "lr": 1e-4,
    "lr_factor": 0.5,
    "lr_patience": 100,
    "stop_patience": 300,
    "stop_min_delta": 0.01,
    "weight_decay": 1e-3,
    "verbose": False,
}

n = 20000
C = np.linspace(0, 0.8, 11)
Txy = np.zeros_like(C)
Tyx = np.zeros_like(C)
for i, c in enumerate(tqdm(C)):
    henon = coupledHenon(n, c)
    X = np.squeeze(henon[:, 0])
    Y = np.squeeze(henon[:, 1])
    ret = cmi(Y, X, u, emb_params, model_params, train_params)
    Txy[i] = ret["cmi_test"]
    ret = cmi(X, Y, u, emb_params, model_params, train_params)
    Tyx[i] = ret["cmi_test"]

# plot
fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
axs[0].set_title("Txy & Tyx")
axs[0].plot(C, Txy, "b", label="Txy")
axs[0].plot(C, Tyx, "r", label="Tyx")
axs[0].legend(loc="lower right")
axs[1].set_title("Txy - Tyx")
axs[1].plot(C, Txy - Tyx, "b")

plt.show()
```

