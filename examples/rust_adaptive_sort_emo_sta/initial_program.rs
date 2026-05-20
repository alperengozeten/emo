// Generic adaptive sorting baseline for the EMO-STA Rust family.

// EVOLVE-BLOCK-START
const INSERTION_SORT_THRESHOLD: usize = 24;

pub fn adaptive_sort<T: Ord + Clone>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }
    quicksort_hybrid(arr);
}

fn quicksort_hybrid<T: Ord + Clone>(arr: &mut [T]) {
    let mut stack = vec![(0usize, arr.len())];

    while let Some((lo, hi)) = stack.pop() {
        if hi <= lo + 1 {
            continue;
        }
        if hi - lo <= INSERTION_SORT_THRESHOLD {
            insertion_sort(&mut arr[lo..hi]);
            continue;
        }

        let pivot_index = lo + choose_pivot_index(&arr[lo..hi]);
        let mid = partition(arr, lo, hi, pivot_index);

        let left = (lo, mid);
        let right = (mid + 1, hi);

        let left_len = left.1.saturating_sub(left.0);
        let right_len = right.1.saturating_sub(right.0);

        if left_len > right_len {
            if left_len > 1 {
                stack.push(left);
            }
            if right_len > 1 {
                stack.push(right);
            }
        } else {
            if right_len > 1 {
                stack.push(right);
            }
            if left_len > 1 {
                stack.push(left);
            }
        }
    }
}

fn choose_pivot_index<T: Ord>(arr: &[T]) -> usize {
    let len = arr.len();
    let last = len - 1;
    let mid = len / 2;

    if arr[0] <= arr[mid] {
        if arr[mid] <= arr[last] {
            mid
        } else if arr[0] <= arr[last] {
            last
        } else {
            0
        }
    } else if arr[0] <= arr[last] {
        0
    } else if arr[mid] <= arr[last] {
        last
    } else {
        mid
    }
}

fn partition<T: Ord + Clone>(arr: &mut [T], lo: usize, hi: usize, pivot_index: usize) -> usize {
    let hi_exclusive = hi;
    let hi_inclusive = hi_exclusive - 1;
    arr.swap(pivot_index, hi_inclusive);
    let pivot = arr[hi_inclusive].clone();

    let mut store = lo;
    for idx in lo..hi_inclusive {
        if arr[idx] < pivot {
            arr.swap(store, idx);
            store += 1;
        }
    }

    arr.swap(store, hi_inclusive);
    store
}

fn insertion_sort<T: Ord>(arr: &mut [T]) {
    for i in 1..arr.len() {
        let mut j = i;
        while j > 0 && arr[j] < arr[j - 1] {
            arr.swap(j, j - 1);
            j -= 1;
        }
    }
}
// EVOLVE-BLOCK-END

#[cfg(test)]
mod tests {
    use super::adaptive_sort;

    #[test]
    fn sorts_basic_values() {
        let mut values = vec![3, 1, 4, 1, 5, 9, 2, 6];
        adaptive_sort(&mut values);
        assert_eq!(values, vec![1, 1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn sorts_duplicates() {
        let mut values = vec![5, 5, 1, 5, 2, 2, 2, 1];
        adaptive_sort(&mut values);
        assert_eq!(values, vec![1, 1, 2, 2, 2, 5, 5, 5]);
    }

    #[test]
    fn handles_reverse_sorted_input() {
        let mut values: Vec<i32> = (0..128).rev().collect();
        adaptive_sort(&mut values);
        assert!(values.windows(2).all(|window| window[0] <= window[1]));
    }
}
