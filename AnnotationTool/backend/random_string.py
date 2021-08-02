# coding=utf8

import random
import string

def get_random_string(length):
    letters = string.ascii_lowercase + "".join([str(i) for i in range(10)]) + "!.~?"
    result_str = ''.join(random.choice(letters) for i in range(length))
    print(result_str)


if __name__ == '__main__':
    get_random_string(10)
