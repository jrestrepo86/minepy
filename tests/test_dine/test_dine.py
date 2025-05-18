from os import name

import plotly.graph_objects as go

from minepy.dine.dine import Dine
from minepy_tests.testTools import simulate_cmi, simulate_mi

N = 1000  # sample size
d = 1  # dimensionality of X and Y
dz = 12  # dimensionality of Z

print("-" * 30)
print("Mutual Information (MI) estimation")
X, Y, Z, mi = simulate_mi(N=N, d=d, rho=0.75, random_state=0)  # here Z is empty
model = Dine(X=X, Y=Y, Z=Z)
model.fit(
    batch_size=64,
    max_epochs=500,
    lr=1e-3,
    stop_patience=500,
    stop_min_delta=0,
    val_size=0.2,
    verbose=True,
)
est = model.get_cmi()
val, cmi_epoch = model.get_curves()

fig1 = go.Figure(data=go.Scatter(x=list(range(len(val))), y=val, name="val"))
fig1.add_trace(go.Scatter(x=list(range(len(cmi_epoch))), y=cmi_epoch, name="mi"))
fig1.show()

print(f"Ground truth: {mi:.4f}")  # 2.5541
print(f"Estimation:   {est:.4f}\n\n")  # 2.5777

print("-" * 30)
print("Conditional Mutual Information (CMI) estimation")
X, Y, Z, cmi = simulate_cmi(N=N, d=d, dz=dz, rho=0.75, random_state=1)
model = Dine(X=X, Y=Y, Z=Z, n_components=16, hidden_sizes=4)
model.fit(
    batch_size=64,
    max_epochs=3000,
    lr=1e-4,
    stop_patience=500,
    stop_min_delta=0,
    val_size=0.2,
    verbose=True,
)
est = model.get_cmi()
val, cmi_epoch = model.get_curves()

fig2 = go.Figure(data=go.Scatter(x=list(range(len(val))), y=val, name="val"))
fig2.add_trace(go.Scatter(x=list(range(len(cmi_epoch))), y=cmi_epoch, name="cmi"))
fig2.show()

print(f"Ground truth: {cmi:.4f}")  # 0.7192
print(f"Estimation:   {est:.4f}\n\n")  # 0.7168
