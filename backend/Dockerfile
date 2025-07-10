FROM docker.1ms.run/nvidia/cuda:12.1.0-base-ubuntu22.04

WORKDIR /app

# 使用中科大源
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    sed -i \
      -e 's|http://archive.ubuntu.com|http://mirrors.ustc.edu.cn|g' \
      -e 's|http://security.ubuntu.com|http://mirrors.ustc.edu.cn|g' \
      /etc/apt/sources.list && \
    # 更新软件包索引
    apt-get update -o Acquire::Check-Valid-Until=false -o Acquire::AllowInsecureRepositories=true

# 安装系统依赖（补充原有缺失组件）
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 编译工具链
    gcc \
    build-essential \
    # 图形渲染库
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgtk2.0-0 \
    # 音视频处理
    ffmpeg \
    # Python构建依赖
    python3-pip \
    python3-setuptools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]