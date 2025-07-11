FROM docker.1ms.run/python:3.11-slim-buster

WORKDIR /app

# 安装Python依赖
COPY ./web/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY ./web ./web
COPY ./backend_common ./backend_common

# 添加公共模块路径
ENV PYTHONPATH="${PYTHONPATH}:/app/backend_common"

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "web.src.main:app", "--host", "0.0.0.0", "--port", "8000"]