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

def rec_wrong_par(str, wronged):
    open = '('
    close = ')'
    option = random.randint(1, 6)
    if option == 1:
        if wronged:
            return '()'
        else:
            return open + rec_wrong_par(str, True)
    elif option == 2:
        return open + rec_wrong_par(str, wronged) + close
    elif option == 3:
        return open + close + rec_wrong_par(str, wronged)
    elif option == 4:
        return rec_wrong_par(str, wronged) + open + close
    elif option == 5:
        return open + rec_wrong_par(str, True)
    else:
        return rec_wrong_par(str, True) + close

def rec_correct_par(str):
    open = '('
    close = ')'
    option = random.randint(1, 4)
    if option == 1:
        return '()'
    elif option == 2:
        return open + rec_correct_par(str) + close
    elif option == 3:
        return open + close + rec_correct_par(str)
    else:
        return rec_correct_par(str) + open + close


def generate_example_parentheses(positive=True):
    if not positive:
        return rec_wrong_par('', False)
    else:
        return rec_correct_par('')
def generate_positive_and_negative(num=500, type='original'):
    if type == 'parenthesis':
        generation = generate_example_parentheses
    else:
        generation = generate_example
    positive = [generation() for i in range(num)]
    negative = [generation(positive=False) for i in range(num)]
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
    pos, neg = generate_positive_and_negative(type='parenthesis')
    c = 3
    # write_examples(pos, neg)