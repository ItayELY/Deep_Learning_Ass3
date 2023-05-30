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

def rec_wrong_par(str, wronged, length):
    open = '('
    close = ')'
    option = random.randint(1, 12)
    if option == 1 and length >= 50:
        if wronged:
            return '()'
        else:
            return open + rec_wrong_par(str, True, length + 1)
    elif option == 2:
        return rec_wrong_par(str, True, length + 1) + close
    elif option == 3:
        return open + close + rec_wrong_par(str, wronged, length + 1)
    elif option == 4:
        return rec_wrong_par(str, wronged, length + 1) + open + close
    elif option == 5:
        return open + rec_wrong_par(str, True, length + 1)
    else:
        return open + rec_wrong_par(str, wronged, length + 1) + close

def rec_correct_par(str, length):
    open = '('
    close = ')'
    option = random.randint(1, 10)
    if option == 1 and length >= 50:
        return '()'
    elif option == 2:
        return rec_correct_par(str, length + 1) + open + close
    elif option == 3:
        return open + close + rec_correct_par(str, length + 1)
    else:
        return open + rec_correct_par(str, length + 1) + close


def generate_example_parentheses(positive=True):
    if not positive:
        return rec_wrong_par('', False, 0)
    else:
        return rec_correct_par('', 0)

def generate_example_an1bn1n2cn2(positive=True):
    if positive:
        n1 = random.randint(20, 40)
        n2 = random.randint(20, 40)
        return 'a' * n1 + 'b' * n1 + 'b' * n2 + 'c' * n2
    else:
        n1 = random.randint(20, 40)
        n2 = random.randint(20, 40)
        n3 = random.randint(40, 80)

        if n3 == n1 + n2:
            while n3 == n1 + n2:
                n3 = random.randint(40, 80)
        return 'a' * n1 + 'b' * n3 + 'c' * n2

def generate_example_an1bnmaxcn2(positive=True):
    if positive:
        n1 = random.randint(20, 100)
        n2 = random.randint(20, 100)
        n3 = random.randint(max(n1, n2) + 1, 101)
        return 'a' * n1 + 'b' * n3 + 'c' * n2
    else:
        n1 = random.randint(20, 40)
        n2 = random.randint(20, 40)
        n3 = random.randint(20, max(n1, n2))
        return 'a' * n1 + 'b' * n3 + 'c' * n2
def generate_example_palindrome(positive=True):
    length = random.randint(50, 100)
    word1 = ''
    for i in range(length):
        c = random.randint(1, 3)
        if c == 1:
            word1 = word1 + 'a'
        elif c == 2:
            word1 = word1 + 'b'
        else:
            word1 = word1 + 'c'
    if positive:
        word1 = word1 + word1[::-1]
    else:
        word2 = ''
        for i in range(length):
            c = random.randint(1, 3)
            if c == 1:
                word2 = word2 + 'a'
            elif c == 2:
                word2 = word2 + 'b'
            else:
                word2 = word2 + 'c'
        word1 = word1 + word2
    return word1

def generate_positive_and_negative(num=500, type='original'):
    if type == 'parenthesis':
        generation = generate_example_parentheses
    elif type == 'an1bn1n2cn2':
        generation = generate_example_an1bn1n2cn2
    elif type == 'palindrome':
        generation = generate_example_palindrome
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