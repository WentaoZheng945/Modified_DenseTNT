import os
import zlib
import pickle

def get_ex_list(file_path):
    pickle_file = open(file_path, 'rb')
    ex_list = pickle.load(pickle_file)
    # self.ex_list = self.ex_list[len(self.ex_list) // 2:]
    pickle_file.close()
    return ex_list

def parse_ex_list(ex_list):
    print('Length of training dataset:', len(ex_list))  # 54926
    for idx, value in enumerate(ex_list):
        if idx == 0:
            return get_idx(ex_list, idx)
        else:
            return None


def get_idx(ex_list, idx):
    # file = self.ex_list[idx]
    # pickle_file = open(file, 'rb')
    # instance = pickle.load(pickle_file)
    # pickle_file.close()

    data_compress = ex_list[idx]
    print('2')
    instance = pickle.loads(zlib.decompress(data_compress))
    return instance

if __name__ == '__main__':
    file_path = r'/home/zwt/thesis/DenseTNT/DenseTNT-argoverse2/argoverse2.densetnt.1/temp_file/ex_list'
    ex_list = get_ex_list(file_path)
    print('1')
    tmp = parse_ex_list(ex_list)
    print(tmp)