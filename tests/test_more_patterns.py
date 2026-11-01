from interview_patterns import more_patterns as m
from interview_patterns.structures import TreeNode


def test_additional_p1_algorithms():
    assert sorted(m.permute_unique([1,1,2]))==sorted([[1,1,2],[1,2,1],[2,1,1]])
    r=m.NextNode(1,m.NextNode(2),m.NextNode(3));assert m.connect_next(r).left.next is r.right
    assert m.gas_station([1,2,3,4,5],[3,4,5,1,2])==3
    head=m.RandomNode(7);head.next=m.RandomNode(13);head.next.random=head
    out=m.copy_random_list(head);assert out is not head and out.next.random is out
    assert m.reconstruct_queue([[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]])==[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
    assert m.split_array_largest_sum([7,2,5,10,8],2)==18
    tree=TreeNode(1,TreeNode(3,TreeNode(5),TreeNode(3)),TreeNode(2,None,TreeNode(9)))
    assert m.largest_values(tree)==[1,3,9]
    assert m.average_of_levels(TreeNode(3,TreeNode(9),TreeNode(20,TreeNode(15),TreeNode(7))))==[3,14.5,11]
    assert m.check_subarray_sum([23,2,4,6,7],6)
    assert not m.check_subarray_sum([1,2,3],7)
    assert m.least_interval(list('AAABBB'),2)==8
    assert abs(m.minmax_gas_dist([1,2,3,4,5,6,7,8,9,10],9)-.5)<1e-5
    assert m.eventual_safe_nodes([[1,2],[2,3],[5],[0],[5],[],[]])==[2,4,5,6]
    assert m.num_subarrays_with_sum([1,0,1,0,1],2)==4
    assert m.subarrays_div_by_k([4,5,0,-2,-3,1],5)==7
    assert m.smallest_divisor([1,2,5,9],6)==5
    assert m.min_days_bouquets([1,10,3,10,2],3,1)==3
    assert m.check_if_prerequisite(3,[[0,1],[1,2]],[[0,2],[2,0]])==[True,False]
    assert m.find_all_recipes(['bread','sandwich'],[['yeast','flour'],['bread','ham']],['yeast','flour','ham'])==['bread','sandwich']
    result=m.sort_items_by_groups(8,2,[-1,-1,1,0,0,1,0,-1],[[],[6],[5],[6],[3,6],[],[],[]])
    assert len(result)==8 and sorted(result)==list(range(8))
    assert m.sort_items_by_groups(2,1,[0,0],[[1],[0]])==[]
