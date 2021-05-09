import pickle
import os

with open(os.path.join('C:\\Users\\허민\\Desktop\\2021-1\\인공지능\\pythonProject2\\mnist\\data\\dataset', 'mlp_digits.pkl'), 'rb') as f:
    mlp = pickle.load(f)

print('딥러닝 데이터 로드 완료')