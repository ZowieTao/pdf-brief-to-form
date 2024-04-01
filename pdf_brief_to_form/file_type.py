from enum import Enum
from typing import Union


class FileType(Enum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"
    UNKNOWN = "unknown"


def get_file_type(file_path: str) -> Union[FileType, str]:
    # 获取文件扩展名
    file_extension = file_path.split(".")[-1].lower()

    # 根据文件扩展名返回对应的 FileType
    if file_extension == "md":
        return FileType.MARKDOWN
    elif file_extension == "pdf":
        return FileType.PDF
    elif file_extension == "docx":
        return FileType.DOCX
    else:
        # 如果扩展名不是上述之一，则返回字符串类型
        return FileType.UNKNOWN
