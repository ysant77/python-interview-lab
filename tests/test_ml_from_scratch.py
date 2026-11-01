import numpy as np
import pytest
from interview_patterns.ml_from_scratch import (
 linear_regression_gd,linear_regression_lstsq,sigmoid_stable,logistic_regression_gd,
 kmeans,knn_predict,layer_norm_numpy,mlp_init,mlp_loss_gradients,mlp_train,mlp_forward,
)
from interview_patterns.tokenization import WordBPETokenizer


def test_linear_regression_solvers():
    x=np.array([[-2.,1.],[-1,2],[0,3],[1,4],[2,5],[3,6]])
    y=3*x[:,0]-2*x[:,1]+5
    fitted=linear_regression_lstsq(x,y)
    np.testing.assert_allclose(fitted.predict(x),y,atol=1e-8)
    stable_x=np.arange(-2,3,dtype=float)[:,None]
    stable_y=2*stable_x[:,0]+1
    result=linear_regression_gd(stable_x,stable_y,lr=.05,steps=500)
    np.testing.assert_allclose(result.predict(stable_x),stable_y,atol=1e-5)
    assert result.losses[-1]<result.losses[0]


def test_logistic_regression_and_stable_sigmoid():
    x=np.array([[-3.,-1.],[-2.,-2.],[2.,2.],[3.,1.]])
    y=np.array([0,0,1,1])
    model=logistic_regression_gd(x,y,steps=400)
    np.testing.assert_array_equal(model.predict(x),y)
    assert model.losses[-1] < model.losses[0]
    z=sigmoid_stable(np.array([-1000.,1000.]))
    np.testing.assert_array_equal(z,[0.,1.])
    with pytest.raises(ValueError):logistic_regression_gd(x,np.array([0,0,2,1]))


def test_kmeans_and_knn():
    x=np.array([[0.,0],[0.1,0],[9.,9],[9.1,9]])
    r=kmeans(x,2,max_iter=30)
    assert set(r.labels)=={0,1}
    assert r.labels[0]==r.labels[1] and r.labels[2]==r.labels[3]
    pred=knn_predict(x,np.array([0,0,1,1]),np.array([[0.2,0],[8.9,8.9]]),k=3)
    np.testing.assert_array_equal(pred,[0,1])
    with pytest.raises(ValueError):knn_predict(x,np.arange(4),x,k=5)


def test_layernorm_matches_torch_reference_when_available():
    x=np.random.default_rng(13).normal(size=(2,3,7))
    result=layer_norm_numpy(x)
    np.testing.assert_allclose(result.mean(axis=-1),0,atol=1e-12)
    torch=pytest.importorskip('torch')
    expected=torch.nn.functional.layer_norm(torch.tensor(x),[7]).numpy()
    np.testing.assert_allclose(result,expected,atol=1e-9)


def test_manual_mlp_backprop_gradient():
    x=np.array([[0.2,-0.7],[0.5,1.2],[-0.3,0.2]])
    y=np.array([0,1,0]);p=mlp_init(2,3,seed=3)
    loss,g=mlp_loss_gradients(x,y,p)
    eps=1e-5
    old=p.w1[0,0]
    p.w1[0,0]=old+eps
    plus=mlp_loss_gradients(x,y,p)[0]
    p.w1[0,0]=old-eps
    minus=mlp_loss_gradients(x,y,p)[0]
    p.w1[0,0]=old
    assert abs(g.w1[0,0]-(plus-minus)/(2*eps))<1e-8
    trained,history=mlp_train(x,y,steps=250,seed=3)
    assert history[-1]<history[0]
    assert mlp_forward(x,trained)[0].shape==(3,)


def test_word_bpe_determinism_and_boundaries():
    b1=WordBPETokenizer(10).fit('low low lower newest widest low')
    b2=WordBPETokenizer(10).fit('low low lower newest widest low')
    assert b1.rules==b2.rules
    assert ''.join(b1.encode_word('lower'))=='lower</w>'
    assert b1.encode('low lower') == [b1.encode_word('low'),b1.encode_word('lower')]
    with pytest.raises(ValueError):WordBPETokenizer(-1)


def test_self_attention_shapes_and_causality():
    torch=pytest.importorskip('torch')
    from interview_patterns.attention import MultiHeadSelfAttention
    torch.manual_seed(11)
    layer=MultiHeadSelfAttention(12,3,causal=True)
    x=torch.randn(2,5,12,requires_grad=True)
    y,w=layer(x,return_weights=True)
    assert y.shape==(2,5,12) and w.shape==(2,3,5,5)
    assert torch.all(w[...,torch.triu(torch.ones(5,5,dtype=torch.bool),diagonal=1)]==0)
    changed=x.detach().clone();changed[:,4,:]+=123
    y2=layer(changed)
    torch.testing.assert_close(y[:,0,:],y2[:,0,:])
    y.square().sum().backward()
    assert x.grad is not None
    with pytest.raises(ValueError):MultiHeadSelfAttention(10,3)
