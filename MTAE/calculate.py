import numpy as np

def calculate_statistics(numbers):
    """
    计算给定数字列表的平均值、方差和标准差。

    参数:
    numbers (list or tuple of float): 包含数字的列表或元组

    返回:
    tuple: (mean, variance, std_dev) 平均值、方差和标准差
    """
    if not numbers or len(numbers) < 2:
        raise ValueError("需要至少两个数字来计算方差和标准差")

    # 使用 numpy 计算平均值、方差和标准差
    mean = np.mean(numbers)
    variance = np.var(numbers, ddof=0)  # 总体方差
    std_dev = np.std(numbers, ddof=0)  # 总体标准差

    # 如果需要样本方差和样本标准差（无偏估计），可以使用以下行代替：
    # variance = np.var(numbers, ddof=1)  # 样本方差
    # std_dev = np.std(numbers, ddof=1)   # 样本标准差

    mean_rounded = round(mean, 4)
    variance_rounded = round(variance, 4)
    std_dev_rounded = round(std_dev, 4)

    return mean_rounded, variance_rounded, std_dev_rounded


# 示例用法
if __name__ == "__main__":
    # 假设你有一系列数字
    numbers = [
        0.5808510638297872,
        0.5886673662119623,
        0.6134453781512605
    ]

    # 调用函数并打印结果
    try:
        mean, variance, std_dev = calculate_statistics(numbers)

        print(f"平均值: {mean}")
        # print(f"方差: {variance}")
        print(f"标准差: {std_dev}")

    except ValueError as e:
        print(e)