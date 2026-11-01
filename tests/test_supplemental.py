from interview_patterns import supplemental_algorithms as a
from interview_patterns import supplemental_more as b
from interview_patterns.structures import ListNode,TreeNode


def test_supplemental_arrays():
    nums=[3,2,2,3];assert a.remove_element(nums,3)==2 and nums[:2]==[2,2]
    assert a.search_rotated_ii([2,5,6,0,0,1,2],0)
    assert not a.search_rotated_ii([2,5,6,0,0,1,2],3)
    assert a.find_min_rotated_ii([2,2,2,0,1])==0
    assert a.is_happy(19) and not a.is_happy(2)
    assert sorted(a.diff_ways_to_compute('2*3-4*5'))==[-34,-14,-10,-10,10]
    assert a.three_sum_smaller([-2,0,1,3],2)==2
    assert a.find_duplicate([1,3,4,2,2])==2
    assert len(a.generate_abbreviations('word'))==16
    assert sorted(a.find_duplicates([4,3,2,7,8,2,3,1]))==[2,3]
    assert a.sequence_reconstruction([1,2,3],[[1,2],[1,3],[2,3]])
    assert not a.sequence_reconstruction([1,2,3],[[1,2],[1,3]])
    assert a.circular_array_loop([2,-1,1,2,2])
    assert not a.circular_array_loop([-1,2])
    assert a.find_maximized_capital(2,0,[1,2,3],[0,1,1])==4
    class Reader:
        def __init__(self,v):self.v=v
        def get(self,i):return self.v[i] if i<len(self.v) else 2**31-1
    assert a.search_unknown_size(Reader([-1,0,3,5,9,12]),9)==4
    assert a.num_subarray_product_less_than_k([10,5,2,6],100)==8
    assert a.next_greatest_letter(['c','f','j'],'j')=='c'
    assert a.employee_free_time([[[1,2],[5,6]],[[1,3]],[[4,10]]])==[[3,4]]
    assert a.flip_and_invert_image([[1,1,0],[1,0,1]])==[[1,0,0],[0,1,0]]
    assert a.backspace_compare('ab#c','ad#c')
    assert not a.backspace_compare('a#c','b')
    assert a.interval_intersection([[0,2],[5,10],[13,23],[24,25]],[[1,5],[8,12],[15,24],[25,26]])==[[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
    assert a.bitwise_complement(5)==2 and a.bitwise_complement(0)==1
    assert a.connect_sticks([2,4,3])==14
    assert a.minimum_abs_difference([4,2,1,3])==[[1,2],[2,3],[3,4]]
    assert a.find_kth_positive([2,3,4,7,11],5)==9
    assert a.min_cost_cut_stick(7,[1,3,4,5])==16
    assert a.find_length_of_shortest_subarray([1,2,3,10,4,2,3,5])==3
    assert a.find_length_of_shortest_subarray([5,4,3,2,1])==4


def test_supplemental_structures():
    head=ListNode(1,ListNode(2,ListNode(3,ListNode(4,ListNode(5)))))
    new=a.rotate_right(head,2)
    assert [new.val,new.next.val,new.next.next.val]==[4,5,1]
    a.reorder_list(new)
    seq=[];cur=new
    while cur:seq.append(cur.val);cur=cur.next
    assert seq==[4,3,5,2,1]
    assert a.num_unique_bsts(3)==5
    assert len(a.unique_bst_trees(3))==5
    tree=TreeNode(3,TreeNode(9),TreeNode(20,TreeNode(15),TreeNode(7)))
    assert a.level_order_bottom(tree)==[[15,7],[9,20],[3]]
    assert a.min_depth(tree)==2
    assert a.sum_numbers(TreeNode(1,TreeNode(2),TreeNode(3)))==25
    assert sorted(a.binary_tree_paths(TreeNode(1,TreeNode(2,None,TreeNode(5)),TreeNode(3))))==['1->2->5','1->3']
    assert a.path_sum_iii(TreeNode(10,TreeNode(5,TreeNode(3,TreeNode(3),TreeNode(-2)),TreeNode(2,None,TreeNode(1))),TreeNode(-3,None,TreeNode(11))),8)==3
    assert a.alien_order(['wrt','wrf','er','ett','rftt'])=='wertf'
    assert a.max_level_sum(TreeNode(1,TreeNode(7,TreeNode(7),TreeNode(-8)),TreeNode(0)))==2


def test_supplemental_ml_style_dp():
    assert b.max_length_ribbons([9,7,5],3)==5
    assert b.fibonacci(10)==55
    assert b.find_closest_elements([1,2,3,4,5],4,3)==[1,2,3,4]
    assert b.is_interleave('aabcc','dbbca','aadbbcbcac')
    assert not b.is_interleave('aabcc','dbbca','aadbbbaccc')
    assert b.largest_subarray([1,4,5,2,3],2)==[5,2]
    assert b.find_least_num_unique_ints([4,3,1,1,3,3,2],3)==2
    assert b.longest_palindrome_subseq('bbbab')==4
    assert b.longest_palindrome('babad') in ('bab','aba')
    assert b.length_of_longest_substring_k_distinct('eceba',2)==3
    assert b.length_of_longest_substring_two_distinct('ccaabbb')==5
    assert b.longest_ones([1,1,1,0,0,0,1,1,1,1,0],2)==6
    assert b.max_taxi_earnings(5,[[2,5,4],[1,5,1]])==7
    assert b.find_median_sorted_arrays([1,3],[2])==2.0
    assert b.find_median_sorted_arrays([1,2],[3,4])==2.5
    assert b.min_insertions_palindrome('mbadm')==2
    assert b.min_cut_palindrome('aab')==1
    assert b.count_palindromic_substrings('aaa')==6
    assert b.minimum_difference([3,9,7,3])==2
    result=b.rearrange_string_k_distance_apart('aabbcc',3)
    assert len(result)==6
    for i,c in enumerate(result):
        assert c not in result[max(0,i-2):i]
    organized=b.reorganize_string('aab');assert organized=='aba'
    assert b.reorganize_string('aaab')==''
    out=b.shortest_common_supersequence('abac','cab')
    assert len(out)==5 and 'abac' in out and 'cab' in out
    assert b.frequency_sort('tree') in ('eert','eetr')
    assert b.find_substring_concat('barfoothefoobarman',['foo','bar'])==[0,9]
    assert b.is_valid_palindrome_k('abcdeca',2)
