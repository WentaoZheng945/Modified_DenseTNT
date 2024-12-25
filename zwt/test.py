import numpy as np
def function1():
    a = {}
    a[0] = {}
    a[1] = {}
    for i in range(2):
        function2(i, a)
    print(a)

def function2(id, a):
    a[id] = {1, 2}

def main():
    a = {'1': [[1, 2], [3, 4]], '2': [[5, 6], [7, 8]]}
    for i in a.keys():
        a[i] = np.array(a[i])
    for i in a.values():
        op(i)
    for i in a.values():
        print(i)

def op(t):
    t[:, 0] -= 1
    t[:, 0] -= 2



if __name__ == '__main__':
    # function1()
    main()
    a = np.array([[1, 2], [3, 4]])
    for i in a:
        print(i)