def function1():
    a = {}
    a[0] = {}
    a[1] = {}
    for i in range(2):
        function2(i, a)
    print(a)

def function2(id, a):
    a[id] = {1, 2}


if __name__ == '__main__':
    function1()