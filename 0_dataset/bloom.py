import math
import os
import pickle
from pybloom_live import (BloomFilter, ScalableBloomFilter)
import signal
import threading
import time
import sys

class BloomFilterThread(threading.Thread):
    '''
    # Properties of bloom filter:
        * false positive rate (p) - probability of false positive
            * e.g. Suppose bloom filter stores `crawled_urls` and we want to check if a `url` is crawled or not and it may say `url_crawled` even though url is not already crawled
            * i.e. we may miss some urls
        * false negative rate - 0
            * e.g. Suppose bloom filter stores `urls` and we want to check if a `url` is already stored and it never says no if the url is already stored
            * i.e. we never re-crawl the same url again
    # Core Features:
        * Back before exiting code (pressing Ctrl + c)
        * load saved on run
        * store huge amount of urls (crawled_urls)
    '''
    def __init__(self, n=2000000, p= 0.00001, save_file='bloom_filter.pkl', scalable=False):
        super().__init__()
        self.save_file = save_file
        self.n = n      # expected number of elements
        self.p = p      # False positive probability
        
        if os.path.exists(save_file):
            # load previously saved bloom filter
            self.load_bloom_filter()
        else:
            if scalable:
                # Scalable bloom filter
                self.bloom = ScalableBloomFilter(mode=ScalableBloomFilter.LARGE_SET_GROWTH, initial_capacity=self.n, error_rate=self.p)
            else:
                # Initialize new bloom filter
                self.bloom = BloomFilter(capacity=n, error_rate=p)

            print('============================================================')
            print(f"\t\t Creaetd new bloom filter")
            print('============================================================', end='\n\n')
        
        self.stop_event = threading.Event()
    
    def __contains__(self, item):
        '''
        e.g. following code checks if a url exists in bloom_filter
        `'https://test.com/' in self.bloom` -> <Bool:True/False>

        '''

        return self.exists(item)
    
    def __len__(self):
        return self.size

    def add(self, item):
        # print('add invoked')
        if type(item)==list:
            # print('add item is list')
            for each_item in item:
                self.bloom.add(each_item)
        else:
            # print('type not list')
            self.bloom.add(item)

    def exists(self, item):
        return item in self.bloom
    
    def size(self):
        '''
        * Returns size in Mb
        '''
        m = (-(self.n * math.log(self.p)) / (math.log(2) ** 2))/(1024**2)
        return int(m)

    def load_bloom_filter(self):
        with open(self.save_file, 'rb') as f:
            self.bloom = pickle.load(f)
        print('============================================================')
        print(f"Bloom filter loaded from file: {self.save_file}")
        print('============================================================', end='\n\n')

    # def run(self):
    #     while not self.stop_event.is_set():
    #         # Simulate some work with the Bloom filter
    #         time.sleep(10000)  # Replace with actual work
    #         # Example: Periodically check for new items to add
    #         # if new_item_to_add:
    #         #     self.add(new_item)
        
    #     # Save the Bloom filter before exiting
    #     self.save_bloom_filter()

    def save_bloom_filter(self):
        with open(self.save_file, 'wb') as f:
            pickle.dump(self.bloom, f)
        print('============================================================')
        print(f"Bloom filter saved to file: {self.save_file}")
        print('============================================================', end='\n\n')
    
    def stop(self):
        self.stop_event.set()
        

def get_bloom_thread(n=2000000, p= 0.00001, save_file='bloom_filter.pkl', scalable=True):
    def signal_handler(sig, frame):
        print('SIGINT received, stopping thread...')
        bloom_thread.save_bloom_filter()
        bloom_thread.stop()
        # bloom_thread.join()
        print('Thread stopped, exiting...')
        sys.exit(0)

    # Register signal handler for SIGINT
    # signal_handler gets invoked while exiting by pressing Ctrl + c
    signal.signal(signal.SIGINT, signal_handler)

    bloom_thread = bloom_thread = BloomFilterThread(scalable=scalable)
    bloom_thread.start()

    return bloom_thread

if __name__ == "__main__":
    n = 20000000  # expected number of elements
    p = 0.01      # false positive probability
    save_file = 'bloom_filter.pkl'
    
    bloom_thread = BloomFilterThread(n, p, save_file)
    bloom_thread.start()

    def signal_handler(sig, frame):
        print('SIGINT received, stopping thread...')
        
        bloom_thread.save_bloom_filter()
        bloom_thread.stop()
        # bloom_thread.join()
        print('Thread stopped, exiting...')
        sys.exit(0)

    # Register signal handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler)

    print(f'exists "http://example.com" : {bloom_thread.exists("http://example.com")}')
    # Simulate adding elements to the Bloom filter
    print(f' adding \'example.com\' {bloom_thread.add("http://example.com")}')
    print(f' adding \'example1.com\' {bloom_thread.add("http://example1.com")}')
    
    # Keep the main thread alive to catch signals
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Keyboard interrupt received, stopping...')
        bloom_thread.stop()
        bloom_thread.join()
        print('Thread stopped, exiting...')




# Using this code from outside this file
'''
from bloom import get_bloom_thread

bloom_thread = get_bloom_thread()

bloom_thread.add('123')
print('123' in bloom_thread)
time.sleep(100)

# press ctrl + c and the bloom_filter is saved
'''