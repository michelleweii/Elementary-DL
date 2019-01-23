from collections import Iterable
import operator

it = iter([1, 2, 3, 4, 5])
print(next(it))
print(next(it))
print(next(it))
print(next(it))
print(next(it))

students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
sorted(students, key=operator.itemgetter(1,2))
print(students)

# sorted_words = [x[0] for x in sorted_word_to_cnt]


