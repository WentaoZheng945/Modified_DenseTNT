import torch

def main(tensor):
    print(tensor.shape)
    for i, each in enumerate(tensor.tolist()):
        print('key:', i)
        # print('value:', each)
        print(type(each))
        if i == 0:
            print(each)


if __name__ == '__main__':
    a = torch.randn(4, 2, 3, 5)
    main(a)
