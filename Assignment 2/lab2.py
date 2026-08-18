import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def loadimg(f):
    i = Image.open(f).convert("L")
    return np.array(i, dtype=np.float64)

def evd(img):
    v, vec = np.linalg.eig(img)
    idx = np.argsort(np.abs(v))[::-1]
    v = v[idx]
    vec = vec[:, idx]
    return vec, np.diag(v), np.linalg.inv(vec), v

def r_evd(vec, v, pinv, k):
    n = len(v)
    dk = np.zeros((n, n), dtype=complex)
    for j in range(k):
        dk[j][j] = v[j]
    res = vec @ dk @ pinv
    return np.real(res)

def svd(img):
    a, b, c = np.linalg.svd(img, full_matrices=False)
    return a, b, c

def r_svd(u, s, vh, k):
    return u[:, :k] @ np.diag(s[:k]) @ vh[:k, :]

def geterr(orig, rec):
    return np.linalg.norm(orig - rec, 'fro')

def plotstuff(orig, rec, k, name):
    eimg = np.abs(orig - rec)
    err = geterr(orig, rec)
    
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.imshow(orig, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(132)
    plt.imshow(np.clip(rec, 0, 255), cmap="gray")
    plt.title(name + " k=" + str(k))
    plt.axis("off")
    
    plt.subplot(133)
    plt.imshow(eimg, cmap="gray")
    plt.title("Error " + str(round(err, 2)))
    plt.axis("off")
    
    plt.show()

img1 = loadimg("cat_10.png")
print("shape of square img:", img1.shape)

if img1.shape[0] != img1.shape[1]:
    print("need square img")
    exit()

vecs, D, pinv, vals = evd(img1)
print("number of eigenvals:", len(vals))

u, s, vh = svd(img1)
print("number of singular vals:", len(s))

kval = [10, 50, 100]

print("EVD results")
err_evd = []
for k in kval:
    rec = r_evd(vecs, vals, pinv, k)
    e = geterr(img1, rec)
    err_evd.append(e)
    print(k, e)
    plotstuff(img1, rec, k, "EVD")

print("SVD results")
err_svd = []
for k in kval:
    rec = r_svd(u, s, vh, k)
    e = geterr(img1, rec)
    err_svd.append(e)
    print(k, e)
    plotstuff(img1, rec, k, "SVD")

mk = min(img1.shape)
all_k = list(range(1, mk + 1))
e1 = []
e2 = []

for k in all_k:
    e1.append(geterr(img1, r_evd(vecs, vals, pinv, k)))
    e2.append(geterr(img1, r_svd(u, s, vh, k)))

plt.plot(all_k, e1, label="EVD")
plt.plot(all_k, e2, label="SVD")
plt.legend()
plt.title("Error vs k")
plt.show()

print("Compare")
for i in range(len(kval)):
    print(kval[i], err_evd[i], err_svd[i])

img2 = loadimg("cat_10 (1).png")
print("Rect image shape:", img2.shape)

u2, s2, vh2 = svd(img2)
m2 = min(img2.shape)
print("max rank", m2)

for k in kval:
    if k <= m2:
        r = r_svd(u2, s2, vh2, k)
        print("SVD for k=", k, " err=", geterr(img2, r))
        plotstuff(img2, r, k, "SVD")

re = []
for k in range(1, m2+1):
    re.append(geterr(img2, r_svd(u2, s2, vh2, k)))

plt.plot(range(1, m2+1), re)
plt.title("SVD Error Rect")
plt.show()