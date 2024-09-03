def calculate_reduction(original_bytes, compressed_bytes):
    """
    计算通信量减少的百分比。
    
    :param original_bytes: 不使用有损压缩时的总通信量（字节数）
    :param compressed_bytes: 使用有损压缩时的总通信量（字节数）
    :return: 减少的百分比
    """
    reduction = original_bytes - compressed_bytes
    reduction_percentage = (reduction / original_bytes) * 100
    return reduction, reduction_percentage

if __name__ == "__main__":
    # 示例数据，单位为字节
    original_total_sent_bytes = 9437184   # 不使用有损压缩时的发送总量
    compressed_total_sent_bytes = 1032921  # 使用有损压缩时的发送总量

    original_total_received_bytes = 9437184   # 不使用有损压缩时的接收总量
    compressed_total_received_bytes = 1032921  # 使用有损压缩时的接收总量

    # 计算发送和接收的减少量及百分比
    sent_reduction, sent_reduction_percentage = calculate_reduction(
        original_total_sent_bytes, compressed_total_sent_bytes
    )

    received_reduction, received_reduction_percentage = calculate_reduction(
        original_total_received_bytes, compressed_total_received_bytes
    )

    # 输出结果
    print(f"Sent bytes reduced by: {sent_reduction} bytes ({sent_reduction_percentage:.2f}%)")
    print(f"Received bytes reduced by: {received_reduction} bytes ({received_reduction_percentage:.2f}%)")
