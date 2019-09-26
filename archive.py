# tools archive

# HOMPACK90 STYLE
def row_dets(mat):
    n, np1 = mat.shape
    qr = np.linalg.qr(mat, 'r')
    dets = np.zeros(np1)
    dets[np1-1] = 1.0
    for lw in range(2, np1+1):
        i = np1 - lw
        ik = i + 1
        dets[i] = -np.dot(qr[i,ik:np1], dets[ik:np1])/qr[i,i]
    dets *= np.sign(np.prod(np.diag(qr))) # just for the sign
    # dets *= np.prod(np.diag(qr)) # to get the scale exactly
    return dets
