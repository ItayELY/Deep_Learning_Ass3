import random

def generate_example(positive=True):
    one = 'a'
    if positive:
        two = 'b'
        three = 'c'
    else:
        two = 'c'
        three = 'b'
    four = 'd'
    to_return = ""
    length = random.randint(1, 10)
    for i in range(length):
        digit = random.randint(1, 9)
        to_return += str(digit)
    length = random.randint(1, 10)
    to_return += one * length

    length = random.randint(1, 10)
    for i in range(length):
        digit = random.randint(1, 9)
        to_return += str(digit)
    length = random.randint(1, 10)
    to_return += two * length

    length = random.randint(1, 10)
    for i in range(length):
        digit = random.randint(1, 9)
        to_return += str(digit)
    length = random.randint(1, 10)
    to_return += three * length

    length = random.randint(1, 10)
    for i in range(length):
        digit = random.randint(1, 9)
        to_return += str(digit)
    length = random.randint(1, 10)
    to_return += four * length

    length = random.randint(1, 10)
    for i in range(length):
        digit = random.randint(1, 9)
        to_return += str(digit)
    return to_return

def generate_positive_and_negative(num=500):
    positive = [generate_example() for i in range(num)]
    negative = [generate_example(positive=False) for i in range(num)]
    return positive, negative


# pos, neg = generate_500_positive_and_negative()
def write_examples(pos, neg, positive_file='./pos_examples', negative_file='./neg_examples'):
    # pos, neg = generate_500_positive_and_negative()
    with open(positive_file, 'w') as f:
        for p in pos:
            f.write(p + '\n')

    with open(negative_file, 'w') as f:
        for n in neg:
            f.write(n + '\n')


def write_to_file(file_name, set):
    with open(file_name, 'w') as f:
        for s in set:
            f.write(s + '\n')
if __name__ == '__main__':
    pos, neg = generate_positive_and_negative()
    write_examples(pos, neg)