import pygam
import os.path
import hashlib

KNOWN_VERSION_TO_FIX = '1aaf0062cafa060d17538db561ecc665'
UPDATED_VERSION_OF_FIX = 'bff6018bf61b852af45b8e034b246fb4'

pygam_path = os.path.abspath(pygam.__file__)
pygam_dir = os.path.dirname(pygam.__file__)
penalties_file_path = os.path.join(pygam_dir, 'penalties.py')
penalties_file_content = open(penalties_file_path, "rb").read()

ans = input('Are you sure you want to install pygam (YES/NO)?')
if ans != "YES":
    print("Exiting without installing...")
    exit()
h = hashlib.md5(penalties_file_content).hexdigest() 
if h != KNOWN_VERSION_TO_FIX:
    if h == UPDATED_VERSION_OF_FIX:
        print("Already updated :-)")
    else:
        print("Error! uknown version of pygam")
    exit()


to_add = b'''

# EDIT by Neural Analysis Project
def circular(n, coef):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes violation of a circular feature function.

   Parameters
    ----------
    n : int
     number of splines
    coef : unused
     for compatibility with constraints

   Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    if n != len(coef.ravel()):
     raise ValueError('dimension mismatch: expected n equals len(coef), '\\
                      'but found n = {}, coef.shape = {}.'\\
                      .format(n, coef.shape))

    if n==1:
    # no first circular penalty for constant functions
        return sp.sparse.csc_matrix(0.)

    row = np.zeros(n)
    row[0] = 1
    row[-1] = -1
    P = sp.sparse.vstack([row, sp.sparse.csc_matrix((n-2, n)), row[::-1]])
    return P.tocsc()

CONSTRAINTS.update({'circular' : circular})
'''
open(penalties_file_path+".bkp", "wb").write(penalties_file_content)
open(penalties_file_path, "wb").write(penalties_file_content + to_add)
print("Fixed GAM! now 'circular constraints' should be available")

