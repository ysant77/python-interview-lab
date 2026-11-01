"""Small deterministic exhaustive/brute-force checks for tricky invariants."""
from itertools import product
from interview_patterns.supplemental_algorithms import (
 find_length_of_shortest_subarray, search_rotated_ii,find_min_rotated_ii,
 num_subarray_product_less_than_k,
)
from interview_patterns.supplemental_more import find_median_sorted_arrays, min_cut_palindrome, reorganize_string
from interview_patterns.catalog_extensions import shortest_subarray_at_least_k,constrained_subset_sum,max_result_jump_game


def test_removal_length_matches_bruteforce():
    for n in range(1,7):
        for nums in product(range(3),repeat=n):
            expected=n
            for i in range(n+1):
                for j in range(i,n+1):
                    cur=nums[:i]+nums[j:]
                    if all(a<=b for a,b in zip(cur,cur[1:])):
                        expected=min(expected,j-i)
            assert find_length_of_shortest_subarray(list(nums))==expected


def test_rotated_binary_search_matches_membership():
    for n in range(1,8):
        for data in product(range(3),repeat=n):
            if list(data)!=sorted(data):continue
            for cut in range(n):
                rotated=list(data[cut:]+data[:cut])
                assert find_min_rotated_ii(rotated)==min(data)
                for target in range(-1,4):
                    assert search_rotated_ii(rotated,target)==(target in data)


def test_shortest_subarray_bruteforce():
    for n in range(1,7):
        for nums in product(range(-2,3),repeat=n):
            for target in (1,3,5):
                candidates=[j-i for i in range(n) for j in range(i+1,n+1) if sum(nums[i:j])>=target]
                expected=min(candidates) if candidates else -1
                assert shortest_subarray_at_least_k(list(nums),target)==expected


def test_palindrome_partition_min_cuts_exhaustive():
    for n in range(1,9):
        for letters in product('ab',repeat=n):
            text=''.join(letters)
            def solve(i):
                if i==len(text):return -1
                return 1+min((solve(j) for j in range(i+1,len(text)+1) if text[i:j]==text[i:j][::-1]))
            assert min_cut_palindrome(text)==solve(0)


def test_reorganization_feasibility_and_adjacent_chars():
    for n in range(1,8):
        for symbols in product('abc',repeat=n):
            text=''.join(symbols);out=reorganize_string(text)
            max_occ=max(text.count(c) for c in set(text))
            feasible=max_occ<=(n+1)//2
            assert bool(out)==feasible
            if feasible:
                assert sorted(out)==sorted(text)
                assert all(x!=y for x,y in zip(out,out[1:]))


def test_median_by_merge():
    for n in range(0,5):
        for m in range(0,5):
            if n+m==0:continue
            for raw in product(range(3),repeat=n+m):
                a=sorted(raw[:n]);b=sorted(raw[n:]);combined=sorted(a+b)
                middle=len(combined)//2
                expected=float(combined[middle]) if len(combined)%2 else (combined[middle-1]+combined[middle])/2
                assert find_median_sorted_arrays(a,b)==expected


def test_product_window_positive():
    for n in range(1,6):
        for nums in product((1,2,3),repeat=n):
            for limit in (1,2,4,8):
                expected=0
                for i in range(n):
                    running=1
                    for j in range(i,n):
                        running*=nums[j]
                        expected+=running<limit
                assert num_subarray_product_less_than_k(list(nums),limit)==expected


def test_max_jump_against_quadratic_dp():
    for n in range(1,8):
        for nums in product((-2,0,3),repeat=n):
            for k in range(1,n+1):
                dp=[nums[0]]
                for j in range(1,n):dp.append(nums[j]+max(dp[max(0,j-k):j]))
                assert max_result_jump_game(list(nums),k)==dp[-1]
