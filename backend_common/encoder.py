import hashlib
# 哈希工具函数
def _hash_password(password: str) -> str:
    """使用 SHA-256 哈希算法处理密码"""
    if not password:
        return None
    return hashlib.sha256(password.encode()).hexdigest()