import yaml

def test_yaml():
    with open('src/configs/stgcn_cnn.yaml') as file:
        documents = yaml.full_load(file)
        print(documents)

if __name__ == '__main__':
    test_yaml()
