import hashlib


def sha1(text: str) -> str:
    """ SHA-1 """
    text = text.encode('utf-8')
    return hashlib.sha1(text).hexdigest()
