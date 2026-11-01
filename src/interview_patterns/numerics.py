"""NumPy exercises with explicit shapes, stability and numerical policies."""
import numpy as np


def pairwise_squared_distances(a:np.ndarray,b:np.ndarray)->np.ndarray:
    """Rows of A (N,D) against rows of B (M,D). O(NMD) time, O(NM) output.
    Broadcasting temporaries may use O(NMD) memory; use blocked matmul for scale.
    """
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:raise ValueError('need (N,D) and (M,D) with equal D')
    return np.sum((a[:,None,:]-b[None,:,:])**2,axis=-1)


def pairwise_cosine_similarity(a:np.ndarray,b:np.ndarray)->np.ndarray:
    """(N,D) @ (D,M) after row normalization. O(NMD), O(NM) output.
    Convention: any comparison involving zero vector returns 0.
    """
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:raise ValueError('incompatible shapes')
    norm_a=np.linalg.norm(a,axis=1,keepdims=True);norm_b=np.linalg.norm(b,axis=1,keepdims=True)
    aa=np.divide(a,norm_a,out=np.zeros_like(a),where=norm_a>0)
    bb=np.divide(b,norm_b,out=np.zeros_like(b),where=norm_b>0)
    return aa@bb.T


def softmax_stable(logits:np.ndarray,axis:int=-1)->np.ndarray:
    """Subtract maximum before exponentiation. O(number of elements) time/space.
    Finite real input only; for infinities use model-specific masking semantics.
    """
    a=np.asarray(logits,dtype=np.float64)
    if not np.all(np.isfinite(a)):raise ValueError('finite logits only')
    ex=np.exp(a-np.max(a,axis=axis,keepdims=True))
    return ex/np.sum(ex,axis=axis,keepdims=True)


def logsumexp_stable(logits:np.ndarray,axis:int=-1)->np.ndarray:
    """Stable log(sum(exp(x))) along axis, finite input. O(n) time/space."""
    a=np.asarray(logits,dtype=np.float64)
    if not np.all(np.isfinite(a)):raise ValueError('finite logits only')
    mx=np.max(a,axis=axis,keepdims=True)
    return np.squeeze(mx+np.log(np.sum(np.exp(a-mx),axis=axis,keepdims=True)),axis=axis)


def prefix_2d(matrix:np.ndarray)->np.ndarray:
    """Integral image with zero top/left border. O(RC) time/space."""
    a=np.asarray(matrix)
    if a.ndim!=2:raise ValueError('2D matrix required')
    p=np.zeros((a.shape[0]+1,a.shape[1]+1),dtype=np.result_type(a,np.int64))
    p[1:,1:]=np.cumsum(np.cumsum(a,axis=0),axis=1)
    return p


def region_sum(prefix:np.ndarray,r1:int,c1:int,r2:int,c2:int):
    """Inclusive 0-indexed rectangle sum. O(1) time, O(1) space."""
    return prefix[r2+1,c2+1]-prefix[r1,c2+1]-prefix[r2+1,c1]+prefix[r1,c1]
