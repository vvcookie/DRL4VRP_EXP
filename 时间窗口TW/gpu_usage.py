#
# import os
# import psutil
#
#
# def get_gpu_mem_info(gpu_id=0):
#     """
#     根据显卡 id 获取显存使用信息, 单位 MB
#     :param gpu_id: 显卡 ID
#     :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
#     """
#     import pynvml
#     pynvml.nvmlInit()
#     if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
#         print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
#         return 0, 0, 0
#
#     handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
#     meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
#     total = round(meminfo.total / 1024 / 1024, 2)
#     used = round(meminfo.used / 1024 / 1024, 2)
#     free = round(meminfo.free / 1024 / 1024, 2)
#     return total, used, free
#
#
# def get_cpu_mem_info():
#     """
#     获取当前机器的内存信息, 单位 MB
#     :return: mem_total 当前机器所有的内存 mem_free 当前机器可用的内存 mem_process_used 当前进程使用的内存
#     """
#     mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
#     mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
#     mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
#     return mem_total, mem_free, mem_process_used
#
#
# if __name__ == "__main__":
#
#     gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=0)
#     print(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'
#           .format(gpu_mem_total, gpu_mem_used, gpu_mem_free))
#
#     cpu_mem_total, cpu_mem_free, cpu_mem_process_used = get_cpu_mem_info()
#     print(r'当前机器内存使用情况：总共 {} MB， 剩余 {} MB, 当前进程使用的内存 {} MB'
#           .format(cpu_mem_total, cpu_mem_free, cpu_mem_process_used))

import torch

torch.cuda.empty_cache()  # 使用memory_allocated前先清空一下cache
allocated=torch.cuda.memory_allocated()
reserved=torch.cuda.memory_reserved()
print(f"已分配显存:{allocated /(10242)} MB")
print(f"保留显存:{reserved /(1024 **2)}MB")

# 目前是6-7-8中间，其次是4-5-6。好像是因为无人机数量过多，就会出发OOM。看来可能和列表有关？
# 6-7 只有选择动作，已经不能在少了 新增16M
# 7-8 新增11M
# 现在又发现5-6之间新增很多……中间是pointer 我哭死