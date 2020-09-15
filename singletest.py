import numpy as np
a=[[1,2],[3,4]]
b=[[5,6],[7,8]]
c=np.concatenate([a,b],-1)
print(c)
c=[a,b]
print(c)
import random

rangek=list(range(0,10))
selectk = random.sample(rangek, 2)
print(rangek)
print(selectk)
oldk=[]
for i in rangek:
	if i not in selectk:
		oldk.append(i)
print(oldk)
from itertools import combinations
com=list(combinations(rangek,2))

print(com)

selectk = random.sample(com, 2)
print(selectk)
oldk=[]
for i in rangek:
	if i not in selectk[0]:
		oldk.append(i)
print(oldk)


import random
print(random.randrange(1, 100))
print(random.randrange(1, 100))
print(random.randrange(1, 100))
print(random.randrange(1, 100))
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA

# #############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
print("mean:",ica.mean_)
print("components:",ica.components_)
# print("S_：",S_)
# print("G:",G)
# print("lda_bar:",lda_bar)
# print("lda_scaling:",lda_scaling)
# print("F:",F)
# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
# pca = PCA(n_components=3,whiten=True)
pca = PCA(n_components=2)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

print("X:",X)
print("mean:",pca.mean_)
print("components:",pca.components_)
print("H:",H)
G=np.dot(X-pca.mean_, pca.components_.T)
print("G:",G)
# plt.figure()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(X)
D=scaler.transform(X)
print("D:",D)
print("mean:",scaler.mean_)
print("scale:",scaler.scale_)
K=(X-scaler.mean_)/scaler.scale_
print("K:",K)





# models = [X, S, S_, H]
# names = ['Observations (mixed signal)',
#          'True Sources',
#          'ICA recovered signals',
#          'PCA recovered signals']
# colors = ['red', 'steelblue', 'orange']

# for ii, (model, name) in enumerate(zip(models, names), 1):
#     plt.subplot(4, 1, ii)
#     plt.title(name)
#     for sig, color in zip(model.T, colors):
#         plt.plot(sig, color=color)

# plt.tight_layout()
# plt.show()