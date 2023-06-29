# Minepy

Mutual information neural estimation


## Install

```bash
git clone https://github.com/jrestrepo86/minepy.git
cd minepy/
pip install -e . 
```

## Usage

### Mine

Mutual information neural estimation:

@inproceedings{belghazi2018mutual,
  title={Mutual information neural estimation},
  author={Belghazi, Mohamed Ishmael and Baratin, Aristide and Rajeshwar, Sai and Ozair, Sherjil and Bengio, Yoshua and Courville, Aaron and Hjelm, Devon},
  booktitle={International conference on machine learning},
  pages={531--540},
  year={2018},
  organization={PMLR}
}

Combating the Instability of Mutual Information-based Losses
via Regularization:

inproceedings{choi2022combating,
  title={Combating the instability of mutual information-based losses via regularization},
  author={Choi, Kwanghee and Lee, Siyeong},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={411--421},
  year={2022},
  organization={PMLR}
}

```py
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from minepy.mine.mine import Mine

def plot(ax, Rho, teo_mi, mi, label):
    ax.plot(Rho, teo_mi, ".k", label="True")
    ax.plot(Rho, mi, "b", label=label)
    ax.legend(loc="upper center")
    ax.set_title(label)


# Net
loss1 = "mine_biased"
loss2 = "mine"
loss3 = "remine"
input_dim = 2
model_params = {"hidden_dim": 50, "afn": "elu", "num_hidden_layers": 3}

mu = np.array([0, 0])
Rho = np.linspace(-0.99, 0.99, 21)
mi_teo = np.zeros(*Rho.shape)
mi_mine_biased = np.zeros(*mi_teo.shape)
mi_mine = np.zeros(*mi_teo.shape)
mi_remine = np.zeros(*mi_teo.shape)

# Training
batch_size = 300
max_epochs = 3000
train_params = {
    "batch_size": batch_size,
    "max_epochs": max_epochs,
    "val_size": 0.2,
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
    model_biased = Mine(input_dim, loss=loss1, **model_params)
    model_mine = Mine(input_dim, loss=loss2, alpha=0.01, **model_params)
    model_remine = Mine(
        input_dim, loss=loss3, regWeight=0.1, targetVal=0, **model_params
    )

    # Train models
    model_biased.fit(X, Z, **train_params)
    model_mine.fit(X, Z, **train_params)
    model_remine.fit(X, Z, **train_params)

    # Get mi estimation
    mi_mine_biased[i] = model_biased.get_mi()
    mi_mine[i] = model_mine.get_mi()
    mi_remine[i] = model_remine.get_mi()

# Plot
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
plot(axs[0], Rho, mi_teo, mi_mine_biased, label=loss1)
plot(axs[1], Rho, mi_teo, mi_mine, label=loss2)
plot(axs[2], Rho, mi_teo, mi_remine, label=loss3)
axs[0].set_xlabel("rho")
axs[1].set_xlabel("rho")
axs[2].set_xlabel("rho")
axs[0].set_ylabel("mi")
fig.suptitle("Mutual information neural estimation", fontsize=13)
plt.show()
```

### Classification based mutual information

CCMI : Classifier based Conditional Mutual Information Estimation

@inproceedings{mukherjee2020ccmi,
  title={CCMI: Classifier based conditional mutual information estimation},
  author={Mukherjee, Sudipto and Asnani, Himanshu and Kannan, Sreeram},
  booktitle={Uncertainty in artificial intelligence},
  pages={1083--1093},
  year={2020},
  organization={PMLR}
}

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
