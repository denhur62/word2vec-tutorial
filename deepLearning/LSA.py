import numpy as np
A = np.array([[0, 0, 0, 1, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0, 1, 0, 0], [
             0, 1, 1, 0, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1, 1]])
np.shape(A)
U, s, VT = np.linalg.svd(A, full_matrices=True)
print(U.round(2))
np.shape(U)
S = np.zeros((4, 9))  # 대각 행렬의 크기인 4 x 9의 임의의 행렬 생성
S[:4, :4] = np.diag(s)  # 특이값을 대각행렬에 삽입
print(S.round(2))
np.shape(S)
A_prime = np.dot(np.dot(U, S), VT)
print(A)
print(A_prime.round(2))
