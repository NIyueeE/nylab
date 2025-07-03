FROM python:3.11-slim-buster

# 安装Python依赖
COPY requirements.txt .
# 使用清华源
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple