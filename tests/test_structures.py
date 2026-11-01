import pytest
from interview_patterns import structures as s, advanced as ad, dynamic_programming as dp


def linked(values):
    head=None
    for x in reversed(values):head=s.ListNode(x,head)
    return head

def values(node,limit=100):
    out=[]
    while node and len(out)<limit:out.append(node.val);node=node.next
    return out

def tree():return s.TreeNode(3,s.TreeNode(9),s.TreeNode(20,s.TreeNode(15),s.TreeNode(7)))


def test_list_cycle():
    h=linked([3,2,0,-4]);tail=h
    while tail.next:tail=tail.next
    tail.next=h.next
    assert s.has_cycle(h) and s.detect_cycle(h) is h.next
    tail.next=None
    assert not s.has_cycle(h) and s.detect_cycle(h) is None
    assert s.middle_node(linked([1,2,3,4])).val==3


def test_list_transforms():
    assert values(s.reverse_list(linked([1,2,3])))==[3,2,1]
    assert values(s.reverse_between(linked([1,2,3,4,5]),2,4))==[1,4,3,2,5]
    assert values(s.reverse_k_group(linked([1,2,3,4,5]),2))==[2,1,4,3,5]
    assert values(s.merge_two_lists(linked([1,2,4]),linked([1,3,4])))==[1,1,2,3,4,4]
    assert values(s.swap_pairs(linked([1,2,3])))==[2,1,3]
    assert values(s.remove_nth_from_end(linked([1,2,3,4,5]),2))==[1,2,3,5]
    assert s.is_palindrome_list(linked([1,2,2,1]))
    shared=linked([7,8]);a=s.ListNode(1,shared);b=s.ListNode(2,shared)
    assert s.get_intersection_node(a,b) is shared


def test_trees():
    root=tree()
    assert s.max_depth(root)==3
    assert s.diameter_of_binary_tree(root)==3
    assert s.level_order(root)==[[3],[9,20],[15,7]]
    assert s.zigzag_level_order(root)==[[3],[20,9],[15,7]]
    assert s.right_side_view(root)==[3,20,7]
    assert s.has_path_sum(root,30)
    assert s.path_sum_ii(root,30)==[[3,20,7]]
    assert s.max_path_sum(root)==47
    bst=s.TreeNode(2,s.TreeNode(1),s.TreeNode(3))
    assert s.is_valid_bst(bst)
    assert s.search_bst(bst,3).val==3
    assert s.kth_smallest_bst(bst,2)==2
    assert s.level_order(s.invert_tree(bst))==[[2],[3,1]]
    assert not s.is_valid_bst(bst)


def test_graphs():
    assert s.num_islands([list('11110'),list('11010'),list('11000'),list('00000')])==1
    assert s.max_area_of_island([[0,1,1],[1,1,0],[0,0,1]])==4
    assert s.flood_fill([[1,1,1],[1,1,0],[1,0,1]],1,1,2)==[[2,2,2],[2,2,0],[2,0,1]]
    assert s.oranges_rotting([[2,1,1],[1,1,0],[0,1,1]])==4
    assert s.can_finish(2,[[1,0]]) and not s.can_finish(2,[[1,0],[0,1]])
    assert s.find_order(2,[[1,0]])==[0,1]
    assert s.find_circle_num([[1,1,0],[1,1,0],[0,0,1]])==2
    assert s.redundant_connection([[1,2],[1,3],[2,3]])==[2,3]
    assert s.network_delay_time([[2,1,1],[2,3,1],[3,4,1]],4,2)==2
    assert s.min_cost_connect_points([[0,0],[2,2],[3,10],[5,2],[7,0]])==20
    assert s.shortest_path_binary_matrix([[0,1],[1,0]])==2
    assert s.closed_island([[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]])==1


def test_trie_and_combinatorics():
    trie=s.Trie();trie.insert('apple')
    assert trie.search('apple') and not trie.search('app') and trie.starts_with('app')
    trie.insert('app');assert trie.search('app')
    assert len(s.subsets([1,2,3]))==8
    assert sorted(s.subsets_with_dup([1,2,2]))==[[],[1],[1,2],[1,2,2],[2],[2,2]]
    assert len(s.permute([1,2,3]))==6
    assert len(s.combine(4,2))==6
    assert sorted(s.combination_sum([2,3,6,7],7))==[[2,2,3],[7]]
    assert s.word_search([list('ABCE'),list('SFCS'),list('ADEE')],'ABCCED')


def test_advanced():
    assert values(ad.merge_k_lists([linked([1,4,5]),linked([1,3,4]),linked([2,6])]))==[1,1,2,3,4,4,5,6]
    assert ad.kth_smallest_sorted_matrix([[1,5,9],[10,11,13],[12,13,15]],8)==13
    assert ad.smallest_range([[4,10,15,24,26],[0,9,12,20],[5,18,22,30]])==[20,24]
    assert ad.find_k_pairs_smallest([1,7,11],[2,4,6],3)==[[1,2],[1,4],[1,6]]
    assert ad.median_sliding_window([1,3,-1,-3,5,3,6,7],3)==[1.0,-1.0,-1.0,3.0,5.0,6.0]
    assert ad.find_right_interval([[3,4],[2,3],[1,2]])==[-1,0,1]
    assert set(ad.single_number_iii([1,2,1,3,2,5]))=={3,5}
    assert set(ad.generate_parenthesis(3))=={'((()))','(()())','(())()','()(())','()()()'}
    assert sorted(ad.letter_case_permutation('a1b'))==['A1B','A1b','a1B','a1b']
    assert sorted(ad.palindrome_partition('aab'))==[['a','a','b'],['aa','b']]
    assert sorted(ad.combination_sum_ii([10,1,2,7,6,1,5],8))==[[1,1,6],[1,2,5],[1,7],[2,6]]
    assert len(ad.solve_n_queens(4))==2
    assert set(ad.min_height_trees(4,[[1,0],[1,2],[1,3]]))=={1}

@pytest.mark.parametrize('fn,args,result',[
    ('climb_stairs',(4,),5),('rob',([1,2,3,1],),4),('rob_circular',([2,3,2],),3),
    ('coin_change',([1,2,5],11),3),('coin_change',([2],3),-1),
    ('coin_change_ii',(5,[1,2,5]),4),('num_squares',(12,),3),
    ('length_of_lis',([10,9,2,5,3,7,101,18],),4),
    ('can_partition',([1,5,11,5],),True),('target_sum',([1,1,1,1,1],3),5),
    ('edit_distance',('horse','ros'),3),
    ('knapsack_01',([1,3,4],[15,20,30],4),35),
    ('unbounded_knapsack',([1,3,4],[15,20,30],4),60),
    ('longest_common_subsequence',('abcde','ace'),3),('unique_paths',(3,7),28),
])
def test_dp(fn,args,result):assert getattr(dp,fn)(*args)==result
