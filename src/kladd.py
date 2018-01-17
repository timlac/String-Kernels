# lst = [['aba', 'b'], ['ca', 'baaaaaa'], ['dda', 'b']]
# bst = [['dda', 'b'],['dda', 'b']]
#
# for i in bst:
#     lst.append(i)
#
# print(lst)
#
# for i in range(2):
#     print(i)


# yx = zip(lst, bst)
#
#
#
# # zip(lst, bst)
# xy =sorted(yx, key=lambda x: len(x[0][0]) + len(x[0][1]), reverse=True)
#
# # print(xy)
#
# l = list(zip(*xy))
#
# # list(l)
#
# lo = list(l[0])
#
# print(lo)
#
# lo.append(["bajs", "kiss"])
#
# print(lo)
#
# # print(lst)
#
# # print(lst[3:])

import numpy as np

T = "hej jag heter tim"

T = np.array(list(T))


tj = np.where(T == "j")[0]

# tj = tj.astype(int)

print(tj)

for i in tj:
    print(i)

g = np.chararray(list(T))

print(g)