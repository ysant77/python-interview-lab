from interview_patterns import catalog_extensions as x
from interview_patterns.structures import TreeNode


def test_backtracking_and_bst():
    assert len(x.letter_combinations('23'))==9
    assert x.total_n_queens(4)==2
    tree=TreeNode(3,TreeNode(1),TreeNode(4,TreeNode(2)))
    x.recover_tree(tree)
    assert tree.val==2 and tree.right.left.val==3
    assert x.single_number_ii([2,2,3,2])==3
    assert x.single_number_ii([-2,-2,1,-2])==1
    word=x.WordDictionary();word.addWord('bad');word.addWord('dad')
    assert word.search('.ad') and not word.search('b..d')
    board=[list('oaan'),list('etae'),list('ihkr'),list('iflv')]
    assert sorted(x.find_words(board,['oath','pea','eat','rain']))==['eat','oath']
    assert board[0]==list('oaan')
    bst=TreeNode(6,TreeNode(2,TreeNode(0),TreeNode(4)),TreeNode(8))
    assert x.lowest_common_ancestor_bst(bst,bst.left,bst.right) is bst
    assert x.count_bits(5)==[0,1,1,2,1,2]
    bst=x.delete_node_bst(bst,2)
    assert bst.left.val==4


def test_trie_extensions():
    assert x.replace_words(['cat','bat','rat'],'the cattle was rattled by the battery')=='the cat was rat by the bat'
    ms=x.MapSum();ms.insert('apple',3);assert ms.sum('ap')==3
    ms.insert('app',2);ms.insert('apple',2);assert ms.sum('ap')==4
    assert x.suggested_products(['mobile','mouse','moneypot','monitor','mousepad'],'mouse')==[
        ['mobile','moneypot','monitor'],['mobile','moneypot','monitor'],
        ['mouse','mousepad'],['mouse','mousepad'],['mouse','mousepad']]


def test_graph_path_and_union_find():
    assert x.swim_in_water([[0,2],[1,3]])==3
    assert x.cheapest_flights(3,[[0,1,100],[1,2,100],[0,2,500]],0,2,1)==200
    assert x.shortest_subarray_at_least_k([2,-1,2],3)==3
    assert x.shortest_subarray_at_least_k([1,2],4)==-1
    assert not x.equations_possible(['a==b','b!=a'])
    assert x.equations_possible(['a==b','b==c','a==c'])
    assert x.maximum_minimum_path([[5,4,5],[1,2,6],[7,4,6]])==4
    assert x.minimum_cost_cities(3,[[1,2,5],[1,3,6],[2,3,1]])==6
    assert x.min_cost_to_supply_water(3,[1,2,2],[[1,2,1],[2,3,1]])==3
    assert x.smallest_string_with_swaps('dcab',[[0,3],[1,2]])=='bacd'
    assert x.make_connected(4,[[0,1],[0,2],[1,2]])==1
    assert x.find_city(4,[[0,1,3],[1,2,1],[1,3,4],[2,3,1]],4)==3
    assert x.max_probability(3,[[0,1],[1,2],[0,2]],[.5,.5,.2],0,2)==.25
    assert x.max_num_edges_to_remove(4,[[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,2],[2,3,4]])==2
    assert x.minimum_effort_path([[1,2,2],[3,8,2],[5,3,5]])==2
    assert x.count_paths(7,[[0,6,7],[0,1,2],[1,2,3],[1,3,3],[6,3,3],[3,5,1],[6,5,1],[2,5,1],[0,4,5],[4,6,2]])==4


def test_range_dp_mst():
    assert x.constrained_subset_sum([10,2,-10,5,20],2)==37
    assert x.max_result_jump_game([1,-1,-2,4,-7,3],2)==7
    assert x.max_value_of_equation([[1,3],[2,0],[5,10],[6,-10]],1)==4
    assert x.is_covered([[1,2],[3,4],[5,6]],2,5)
    assert x.split_painting([[1,4,5],[4,7,7],[1,7,9]])==[[1,4,14],[4,7,16]]
    assert x.shifting_letters('abc',[[0,1,0],[1,2,1],[0,2,1]])=='ace'
    assert x.find_critical_and_pseudo_critical_edges(5,[[0,1,1],[1,2,1],[2,3,2],[0,3,2],[0,4,3],[3,4,3],[1,4,6]])==[[0,1],[2,3,4,5]]
