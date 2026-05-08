# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot

class K_Means(object):
    """
    一个实现 K-Means 聚类算法的类。

    该算法通过迭代的方式将数据点划分到 K 个不同的簇中，使得每个数据点都属于
    与其最近的均值（簇中心）对应的簇。

    Attributes:
        k (int): 聚类的目标簇数。
        tolerance (float): 用于判断聚类中心是否收敛的阈值。当所有簇中心的移动距离平方和小于此值时，迭代停止。
        max_iter (int): 最大迭代次数，防止无限循环。
        centers (dict): 存储最终每个簇的中心点坐标。键是簇的索引(0 to k-1)，值是中心点的 numpy 数组。
        clf (dict): 存储最终的聚类结果。键是簇的索引，值是该簇包含的所有数据点的列表。
    """

    def __init__(self, k: int = 2, tolerance: float = 1e-4, max_iter: int = 300):
        """
        初始化 K-Means 聚类器。

        Args:
            k (int, optional): 聚类的簇数。默认为 2。
            tolerance (float, optional): 收敛阈值。默认为 0.0001。
            max_iter (int, optional): 最大迭代次数。默认为 300。
        """
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.centers = {}
        self.clf = {}

    def fit(self, data: np.ndarray):
        """
        使用输入数据来训练 K-Means 模型，即找到 K 个簇的中心。

        Args:
            data (np.ndarray): 训练数据集，形状应为 (n_samples, n_features)。
        """
        # 1. 初始化簇中心
        #    - 选择数据集中的前 k 个点作为初始中心点。
        #    - 注意：这是一种简单的初始化方法，更稳健的方法是随机选择k个点。
        #    - 原始代码中的 `+ 1e-6` 是为了防止后续计算中分母为0，但在新的收敛判断下已非必需。
        self.centers = {i: data[i] for i in range(self.k)}

        # 2. 开始迭代优化
        for i in range(self.max_iter):
            # 清空上一轮的分类结果
            self.clf = {j: [] for j in range(self.k)}

            # --- 优化点：向量化计算距离，分配样本到最近的簇 ---
            # 将中心点字典转换为 (k, n_features) 的 numpy 数组，方便计算
            center_array = np.array(list(self.centers.values()))

            # 使用 numpy广播机制 计算所有数据点到所有中心点的距离的平方
            # 扩展维度: data -> (n_samples, 1, n_features)
            #           center_array -> (1, k, n_features)
            # 广播后相减得到 (n_samples, k, n_features) 的差异矩阵
            # 沿特征轴求和得到 (n_samples, k) 的距离平方矩阵
            distances_sq = np.sum((data[:, np.newaxis, :] - center_array[np.newaxis, :, :]) ** 2, axis=2)

            # 找到每个数据点距离最近的中心的索引（即标签）
            labels = np.argmin(distances_sq, axis=1)

            # 根据标签将数据点分配到对应的簇
            for idx, label in enumerate(labels):
                self.clf[label].append(data[idx])
            # --- 向量化计算结束 ---

            # 3. 重新计算每个簇的中心点
            #    - 新的中心点是簇内所有数据点的平均值。
            prev_centers = dict(self.centers)  # 保存旧的中心点用于后续收敛判断
            for c in self.clf:
                # 仅在簇不为空时更新，避免空簇导致错误
                if self.clf[c]:
                    self.centers[c] = np.average(self.clf[c], axis=0)

            # 4. 判断中心点是否收敛
            #    - 检查新旧中心点的移动距离是否小于阈值。
            #    - 优化点：使用更稳健的“中心点移动距离平方和”作为判断标准。
            is_optimized = True
            total_shift = 0
            for center_idx in self.centers:
                original_center = prev_centers[center_idx]
                current_center = self.centers[center_idx]
                total_shift += np.sum((current_center - original_center) ** 2)

            if total_shift < self.tolerance:
                is_optimized = True
                break
            else:
                is_optimized = False

        # 将最终的簇成员列表转换为numpy数组
        for c in self.clf:
            self.clf[c] = np.array(self.clf[c])

    def predict(self, p_data: np.ndarray) -> int:
        """
        预测单个数据点所属的簇。

        Args:
            p_data (np.ndarray): 需要预测的数据点，形状为 (n_features,).

        Returns:
            int: 预测的簇索引。
        """
        # 计算该点到所有簇中心的欧氏距离
        distances = [np.linalg.norm(p_data - self.centers[center_idx]) for center_idx in self.centers]
        # 返回距离最小的那个簇的索引
        return distances.index(min(distances))

    def classify_dataset(self, raw_data: np.ndarray) -> np.ndarray:
        """
        对整个数据集进行分类，返回每个数据点的簇标签。

        Args:
            raw_data (np.ndarray): 需要分类的数据集。

        Returns:
            np.ndarray: 包含每个数据点标签的一维数组。
        """
        # 使用列表推导式高效地对每个数据点进行预测
        return np.array([self.predict(data_point) for data_point in raw_data])

    def classify_dataset_find_N_Max(self, raw_data: np.ndarray, find_max_n: int = 10,
                                    sort_by_feature_index: int = 4) -> tuple:
        """
        对数据集进行分类，并为每个簇找到距离其中心最近的 N 个样本。
        同时，根据簇中心某个特征的值对簇进行排序，可用于解释簇的含义（如保守 vs 激进）。

        Args:
            raw_data (np.ndarray): 需要分类的数据集。
            find_max_n (int, optional): 每个簇最多寻找的最近样本数量。默认为 10。
            sort_by_feature_index (int, optional): 用于对簇中心进行排序的特征列索引。默认为 4。

        Returns:
            tuple: 包含三个元素的元组:
                - (list[np.ndarray]): 一个列表，每个元素是一个簇的最近N个样本的**原始索引**。
                - (np.ndarray): 所有数据点的簇标签。
                - (np.ndarray): 簇中心的排序索引（例如，按某个特征从低到高排序）。
        """
        # 1. 对所有数据点进行分类，得到它们的标签
        all_labels = self.classify_dataset(raw_data)

        # 2. 为每个簇找到最近的 N 个样本
        top_samples_per_cluster = []
        for i in range(self.k):
            # 找到属于当前簇的所有数据点的原始索引
            indices_in_cluster = np.where(all_labels == i)[0]

            if len(indices_in_cluster) == 0:
                top_samples_per_cluster.append(np.array([], dtype=int))
                continue

            # 获取这些数据点
            points_in_cluster = raw_data[indices_in_cluster]

            # 计算这些点到其簇中心的距离
            distances = np.linalg.norm(points_in_cluster - self.centers[i], axis=1)

            # 根据距离排序，得到局部索引
            sorted_local_indices = np.argsort(distances)

            # 选取前 N 个（如果数量不足，则全选）
            top_n_local_indices = sorted_local_indices[:find_max_n]

            # 将局部索引映射回原始数据索引
            top_n_global_indices = indices_in_cluster[top_n_local_indices]
            top_samples_per_cluster.append(top_n_global_indices)

        # 3. 根据特定特征对簇中心进行排序
        #    - 这可以用来给簇赋予业务含义，例如，如果第4个特征是“平均速度”，
        #      排序后就可以得到“低速簇”、“中速簇”、“高速簇”的索引。
        center_values = np.array([self.centers[i][sort_by_feature_index] for i in range(self.k)])
        sorted_center_indices = np.argsort(center_values)  # 升序排序，得到索引

        return top_samples_per_cluster, all_labels, sorted_center_indices

if __name__ == '__main__':
    # --- 示例代码 ---
    # 创建一个示例数据集
    x = np.array([[1, 2, 4], [1.5, 1.8, 1.6], [5, 8, 6], [8, 8, 7], [1, 0.6, 0.8], [9, 11, 10]])

    # 初始化 K-Means 模型，设定 k=3
    k_means = K_Means(k=3)

    # 训练模型
    k_means.fit(x)

    # 打印最终的簇中心
    print("最终的簇中心点:")
    print(k_means.centers)

    # 可视化聚类结果
    colors = ['r', 'g', 'b']

    # 绘制每个簇的数据点
    for cat in k_means.clf:
        for point in k_means.clf[cat]:
            pyplot.scatter(point[0], point[1], c=colors[cat])

    # 绘制簇中心点（用星号表示）
    for center_idx in k_means.centers:
        center_point = k_means.centers[center_idx]
        pyplot.scatter(center_point[0], center_point[1], c=colors[center_idx], marker='*', s=150, edgecolor='black')

    pyplot.title("K-Means Clustering Result")
    pyplot.xlabel("Feature 1")
    pyplot.ylabel("Feature 2")
    pyplot.grid(True)
    pyplot.show()