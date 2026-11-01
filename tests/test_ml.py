import pytest
np=pytest.importorskip('numpy')
from interview_patterns.numerics import pairwise_squared_distances,pairwise_cosine_similarity,softmax_stable,logsumexp_stable,prefix_2d,region_sum

def test_pairwise_distances():
    a=np.array([[0,0],[1,1]]);b=np.array([[1,0],[0,2]])
    np.testing.assert_allclose(pairwise_squared_distances(a,b),[[1,4],[1,2]])
    with pytest.raises(ValueError):pairwise_squared_distances(np.ones((3,2)),np.ones((2,3)))

def test_cosine_similarity_with_zero_vector():
    a=np.array([[1,0],[0,0]]);b=np.array([[1,0],[0,1]])
    np.testing.assert_allclose(pairwise_cosine_similarity(a,b),[[1,0],[0,0]])

def test_numerical_stability():
    logits=np.array([[1000.0,1001.0,1002.0],[-1000.0,-999.0,-998.0]])
    softmax=softmax_stable(logits)
    np.testing.assert_allclose(softmax.sum(axis=-1),[1,1])
    assert np.all(np.isfinite(softmax))
    np.testing.assert_allclose(logsumexp_stable(logits),
        np.array([1002.0,-998.0])+np.log(np.exp([-2.,-1.,0.]).sum()))

def test_integral_image():
    p=prefix_2d(np.array([[1,2,3],[4,5,6],[7,8,9]]))
    assert p.shape==(4,4)
    assert region_sum(p,1,1,2,2)==28

@pytest.mark.slow
def test_torch_drills(tmp_path):
    torch=pytest.importorskip('torch')
    from interview_patterns.torch_drills import gradient_check,tiny_train_eval,attention_shape_demo,amp_profile_demo
    analytical,finite=gradient_check()
    assert abs(analytical-finite)<1e-6
    metrics=tiny_train_eval(epochs=40,save_path=str(tmp_path/'weights.pt'))
    assert metrics['held_out_mse']<1.0
    shapes=attention_shape_demo(batch=2,heads=3,length=5,dim=4)
    assert shapes['query']==(2,3,5,4) and shapes['naive_score_elements']==150
    info=amp_profile_demo(steps=1)
    assert info['profile_events']>0


def test_lora_freezing_and_learning():
    pytest.importorskip('torch')
    from interview_patterns.lora_toy import demo_step
    result=demo_step(steps=80)
    assert result['base_frozen'] and result['trainable_parameters']==(2*8+6*2)
    assert result['final_mse']<result['initial_mse']
