# 命令后可加数字表示选几个 GPU, 不加则默认为 1

# 用法 1 
# . ./set_cuda.sh && python test.py  # 每次执行都会更新可用 GPU, 最前面的点可以换成 source

# 用法 2
# 在本文件最后加入要执行的命令: python test.py, 可以改名字后直接执行该脚本


# 测试 1 (With Tensorflow)
# . ./set_cuda.sh 3 && python -c 'import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; import tensorflow as tf; [print(gpu) for gpu in tf.config.list_physical_devices("GPU")]'

# 测试 2 (With Pytorch)
# . ./set_cuda.sh 2 && python -c 'import torch; [print(torch.cuda.get_device_name(idx)) for idx in range(torch.cuda.device_count())]'  

#########################################################

# 按个数获取剩余 RAM 最大的 GPU, 返回可见的 GPU 序列
get_cuda_by_memory(){
  unset CUDA_VISIBLE_DEVICES  # 清空原来的环境变量
  nvidia-smi  --query-gpu=index,memory.free --format=csv,noheader | sort -n -k 2 | tail -n $1 | awk -F, '{print $1}' | xargs | sed 's/ /,/g' 
}

# 可见的 GPU 数量, 默认为 1
gpu_nums=${1:-1}
# 调用函数, 设置可见的 GPU 个数, 默认为 1
gpus=$(get_cuda_by_memory $gpu_nums)
echo -------------------------------------------------
echo "Using GPU --> [$gpus]; Here are the details ↓↓↓ "
nvidia-smi -i $gpus   --query-gpu=index,gpu_name,utilization.memory,memory.free --format=csv
echo -------------------------------------------------
# 正式设置可见 GPU
export CUDA_VISIBLE_DEVICES=$gpus

# 已知 bug: DGX 上的 GPU-4 的实际编号用 3, 而 nvidia-smi 给它的编号为 4

#####################################################################

# 可直接在这里加自己的命令
# python main.py
