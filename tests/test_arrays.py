from collections import Counter
from itertools import product
import random
import pytest
from interview_patterns import arrays as a

@pytest.mark.parametrize('fn,args,expected',[
    ('two_sum',([2,7,11,15],9),[0,1]),
    ('valid_anagram',('anagram','nagaram'),True),
    ('valid_anagram',('a','ab'),False),
    ('contains_duplicate',([1,2,3,1],),True),
    ('contains_duplicate',([],),False),
    ('longest_unique_substring',('abcabcbb',),3),
    ('longest_unique_substring',('',),0),
    ('min_subarray_len',(7,[2,3,1,2,4,3]),2),
    ('character_replacement',('AABABBA',1),4),
    ('contains_permutation',('ab','eidbaooo'),True),
    ('contains_permutation',('ab','eidboaoo'),False),
    ('total_fruit',([1,2,1],),3),
    ('min_window',('ADOBECODEBANC','ABC'),'BANC'),
    ('min_window',('a','aa'),''),
    ('max_area',([1,8,6,2,5,4,8,3,7],),49),
    ('two_sum_sorted',([2,7,11,15],9),[1,2]),
    ('three_sum_closest',([-1,2,1,-4],1),2),
    ('trap_rain_water',([4,2,0,3,2,5],),9),
    ('is_palindrome',('A man, a plan, a canal: Panama',),True),
    ('product_except_self',([1,2,3,4],),[24,12,8,6]),
    ('product_except_self',([0,1,0],),[0,0,0]),
    ('binary_search',([-1,0,3,5,9,12],9),4),
    ('binary_search',([1,2,3],4),-1),
    ('search_rotated',([4,5,6,7,0,1,2],0),4),
    ('search_range',([5,7,7,8,8,10],8),[3,4]),
    ('search_insert',([1,3,5,6],2),1),
    ('find_min_rotated',([3,4,5,1,2],),1),
    ('min_eating_speed',([3,6,7,11],8),4),
    ('ship_within_days',([1,2,3,4,5,6,7,8,9,10],5),15),
    ('count_subarray_sum',([1,1,1],2),2),
    ('count_subarray_sum',([1,-1,0],0),3),
    ('pivot_index',([1,7,3,6,5,6],),3),
    ('next_greater_element',([4,1,2],[1,3,4,2]),[-1,3,-1]),
    ('daily_temperatures',([73,74,75,71,69,72,76,73],),[1,1,4,2,1,1,0,0]),
    ('largest_rectangle_area',([2,1,5,6,2,3],),10),
    ('max_sliding_window',([1,3,-1,-3,5,3,6,7],3),[3,3,5,5,6,7]),
    ('longest_subarray_limit',([8,2,4,7],4),2),
    ('merge_intervals',([[1,3],[2,6],[8,10],[15,18]],),[[1,6],[8,10],[15,18]]),
    ('insert_interval',([[1,3],[6,9]],[2,5]),[[1,5],[6,9]]),
    ('can_attend_meetings',([[0,30],[5,10],[15,20]],),False),
    ('can_attend_meetings',([[0,5],[5,9]],),True),
    ('min_meeting_rooms',([[0,30],[5,10],[15,20]],),2),
    ('jump_game',([2,3,1,1,4],),True),
    ('jump_game',([3,2,1,0,4],),False),
    ('jump_game_ii',([2,3,1,1,4],),2),
    ('partition_labels',('ababcbacadefegdehijhklij',),[9,7,8]),
    ('valid_parentheses',('([]){}',),True),
    ('valid_parentheses',('([)]',),False),
    ('missing_number',([3,0,1],),2),
    ('single_number',([4,1,2,1,2],),4),
    ('hamming_weight',(11,),3),
    ('range_addition',(5,[[1,3,2],[2,4,3],[0,2,-2]]),[-2,0,3,5,3]),
    ('car_pooling',([[2,1,5],[3,3,7]],4),False),
    ('corp_flight_bookings',([[1,2,10],[2,3,20],[2,5,25]],5),[10,55,45,25,25]),
    ('first_missing_positive',([3,4,-1,1],),2),
    ('find_disappeared_numbers',([4,3,2,7,8,2,3,1],),[5,6]),
    ('sorted_squares',([-4,-1,0,3,10],),[0,1,9,16,100]),
    ('find_anagrams',('cbaebabacd','abc'),[0,6]),
    ('max_average_subarray',([1,12,-5,-6,50,3],4),12.75),
    ('reverse_bits',(0b00000000000000000000000000000001,),2147483648),
    ('maximal_rectangle',([['1','0','1','0','0'],['1','0','1','1','1'],['1','1','1','1','1'],['1','0','0','1','0']],),6),
    ('erase_overlap_intervals',([[1,2],[2,3],[3,4],[1,3]],),1),
    ('find_min_arrow_shots',([[10,16],[2,8],[1,6],[7,12]],),2),
    ('next_greater_elements_circular',([1,2,1],),[2,-1,2]),
    ('top_k_frequent_words',(['i','love','leetcode','i','love','coding'],2),['i','love']),
    ('last_stone_weight',([2,7,4,1,8,1],),1),
])
def test_solutions(fn,args,expected):
    assert getattr(a,fn)(*args)==expected

def test_three_sum():
    assert sorted(a.three_sum([-1,0,1,2,-1,-4]))==[[-1,-1,2],[-1,0,1]]

def test_four_sum():
    from interview_patterns.advanced import four_sum
    assert sorted(four_sum([1,0,-1,0,-2,2],0))==[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]

def test_top_k_frequent_ties():
    vals=a.top_k_frequent([1,1,1,2,2,3],2)
    assert set(vals)=={1,2}

def test_k_closest():
    assert a.k_closest([[1,3],[-2,2]],1)==[[-2,2]]

def test_sort_colors_and_duplicate_removal():
    nums=[2,0,2,1,1,0];a.sort_colors(nums);assert nums==[0,0,1,1,2,2]
    nums=[1,1,2];k=a.remove_duplicates_sorted(nums);assert k==2 and nums[:k]==[1,2]

def test_exhaustive_short_inputs_against_brute_force():
    for length in range(5):
        for tup in product([0,1,2],repeat=length):
            nums=list(tup)
            assert a.longest_unique_substring(''.join(map(str,nums))) == max([j-i for i in range(length+1) for j in range(i,length+1) if len(set(nums[i:j]))==j-i] or [0])
            for target in range(4):
                assert a.count_subarray_sum(nums,target)==sum(sum(nums[i:j])==target for i in range(length) for j in range(i+1,length+1))

def test_large_sliding_window_against_brute_force():
    rng=random.Random(37)
    for _ in range(100):
        nums=[rng.randrange(-6,7) for _ in range(rng.randrange(1,20))]
        k=rng.randrange(1,len(nums)+1)
        assert a.max_sliding_window(nums,k)==[max(nums[i:i+k]) for i in range(len(nums)-k+1)]
