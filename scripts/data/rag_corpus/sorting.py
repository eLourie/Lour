"""Reference sorting algorithms for the RAG code-loader demo."""


def quicksort(items: list[int]) -> list[int]:
    """Quicksort: divide-and-conquer, average O(n log n) time complexity."""
    if len(items) <= 1:
        return items
    pivot = items[len(items) // 2]
    left = [x for x in items if x < pivot]
    middle = [x for x in items if x == pivot]
    right = [x for x in items if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def bubble_sort(items: list[int]) -> list[int]:
    """Bubble sort: simple O(n^2) comparison sort, useful only for teaching."""
    data = list(items)
    for i in range(len(data)):
        for j in range(len(data) - i - 1):
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return data
