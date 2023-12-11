import ray
import random
import os
import numpy as np
import itertools
import time
import json

bucket_size_threshold = 50000 # only actually sample sort if data length larger than this 

def timeable_function(func):
    def wrapper(*args, **kwargs):
        print(f">> timer start [{func.__name__}] <<")
        start = time.perf_counter_ns()
        out = func(*args, **kwargs)
        end = time.perf_counter_ns()
        print(f">> timer end [{func.__name__}]: {(end-start)/1e6}ms <<")
        return out
    
    return wrapper

@timeable_function
def load_data():
    f = open("random_numbers.txt", "r")
    data = [float(l.strip()) for l in f.readlines()]
    f.close()
    return data

def ss_generate_sample(data, num_samples):
    sample_size = len(data) // num_samples
    start_indices = random.sample(range(len(data) - sample_size + 1), num_samples)
    samples = [data[i:i + sample_size] for i in start_indices]
    return sorted([elem for sample in samples for elem in sample])

def ss_choose_splitters(sample, num_samples):
    percentiles = np.linspace(0, 100, num_samples + 1)
    splitters = np.percentile(sample, percentiles)
    return np.round(splitters).astype(int)

def ss_build_buckets(data, splitters):
    buckets = [[] for _ in range(len(splitters))]
    for datum in data:
        for i, splitter in enumerate(splitters):
            if datum <= splitter:
                buckets[i].append(datum)
                break
        else:
            buckets[-1].append(datum)
    return buckets

@ray.remote
def sample_sort(data, num_buckets):
    if len(data) / num_buckets < bucket_size_threshold:
        return sorted(data)

    num_samples = num_buckets - 1
    sample = ss_generate_sample(data, num_samples)
    splitters = ss_choose_splitters(sample, num_samples)
    buckets = ss_build_buckets(data, splitters)

    return list(itertools.chain.from_iterable(ray.get([sample_sort.remote(bucket, num_buckets) for bucket in buckets])))

@timeable_function
def timed_sample_sort(data, num_buckets):
    return ray.get(sample_sort.remote(data, num_buckets))

@timeable_function
def timed_native_sort(data):
    return sorted(data)

def main():
    print("Initializing ray...")
    ctx = ray.init()
    
    print("Loading unsorted data...")
    data = load_data()
    print(f"...loaded {len(data)} numbers...")
    
    print("Sorting with native sorted() function...")
    native = timed_native_sort(data)

    print("Sorting with sample_sort() function...")
    sampled = timed_sample_sort(data, os.cpu_count())

    # If this throws an AssertionError, my algorithm is bad. If nothing, we're all good.
    assert native == sampled, "Native sort does not match the sample sort output (sample sort algorithm invalid)"

    print("Sleeping 600 seconds for time to review ray dashboard data...")
    time.sleep(600) # time to review the ray dashboard if you want

if __name__ == "__main__":
    main()