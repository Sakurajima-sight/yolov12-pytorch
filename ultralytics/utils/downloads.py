# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import re
import shutil
import subprocess
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from urllib import parse, request

import requests
import torch

from ultralytics.utils import LOGGER, TQDM, checks, clean_url, emojis, is_online, url2file

# 定义 Ultralytics GitHub 的资产资源仓库，维护在 https://github.com/ultralytics/assets
GITHUB_ASSETS_REPO = "ultralytics/assets"
GITHUB_ASSETS_NAMES = (
    [f"yolov8{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose", "-obb", "-oiv7")]
    + [f"yolo11{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose", "-obb")]
    + [f"yolov5{k}{resolution}u.pt" for k in "nsmlx" for resolution in ("", "6")]
    + [f"yolov3{k}u.pt" for k in ("", "-spp", "-tiny")]
    + [f"yolov8{k}-world.pt" for k in "smlx"]
    + [f"yolov8{k}-worldv2.pt" for k in "smlx"]
    + [f"yolov9{k}.pt" for k in "tsmce"]
    + [f"yolov10{k}.pt" for k in "nsmblx"]
    + [f"yolo_nas_{k}.pt" for k in "sml"]
    + [f"sam_{k}.pt" for k in "bl"]
    + [f"FastSAM-{k}.pt" for k in "sx"]
    + [f"rtdetr-{k}.pt" for k in "lx"]
    + ["mobile_sam.pt"]
    + ["calibration_image_sample_data_20x128x128x3_float32.npy.zip"]
)
GITHUB_ASSETS_STEMS = [Path(k).stem for k in GITHUB_ASSETS_NAMES]


def is_url(url, check=False):
    """
    判断一个字符串是否为 URL，并可选地检查该 URL 是否可访问。

    参数:
        url (str): 要验证的 URL 字符串。
        check (bool, 可选): 如果为 True，则额外检查该 URL 是否在线可访问。
            默认为 False。

    返回:
        (bool): 如果是有效的 URL，则返回 True。如果设置了 check，则返回 URL 在线可访问的判断结果。
            否则返回 False。

    示例:
        ```python
        valid = is_url("https://www.example.com")
        ```
    """
    try:
        url = str(url)
        result = parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # 检查是否为 URL
        if check:
            with request.urlopen(url) as response:
                return response.getcode() == 200  # 检查是否在线存在
        return True
    except Exception:
        return False


def delete_dsstore(path, files_to_delete=(".DS_Store", "__MACOSX")):
    """
    删除指定目录下所有 ".DS_Store" 文件。

    参数:
        path (str, 可选): 要删除 ".DS_Store" 文件的目录路径。
        files_to_delete (tuple): 要删除的文件名。

    示例:
        ```python
        from ultralytics.utils.downloads import delete_dsstore

        delete_dsstore("path/to/dir")
        ```

    注意:
        ".DS_Store" 文件是苹果操作系统创建的，用于存储文件夹和文件的元数据。
        它们是隐藏的系统文件，在跨平台传输文件时可能会引起问题。
    """
    for file in files_to_delete:
        matches = list(Path(path).rglob(file))
        LOGGER.info(f"正在删除 {file} 文件: {matches}")
        for f in matches:
            f.unlink()


def zip_directory(directory, compress=True, exclude=(".DS_Store", "__MACOSX"), progress=True):
    """
    压缩目录内容为 zip 文件，排除掉文件名包含 exclude 中字符串的文件。生成的 zip 文件将和目录同名，并保存在同一目录下。

    参数:
        directory (str | Path): 要压缩的目录路径。
        compress (bool): 是否进行压缩，默认为 True。
        exclude (tuple, 可选): 要排除的文件名字符串元组。默认为 ('.DS_Store', '__MACOSX')。
        progress (bool, 可选): 是否显示进度条，默认为 True。

    返回:
        (Path): 返回生成的 zip 文件的路径。

    示例:
        ```python
        from ultralytics.utils.downloads import zip_directory

        file = zip_directory("path/to/dir")
        ```
    """
    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

    delete_dsstore(directory)
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"目录 '{directory}' 不存在。")

    # 生成待压缩文件列表，并使用进度条进行压缩
    files_to_zip = [f for f in directory.rglob("*") if f.is_file() and all(x not in f.name for x in exclude)]
    zip_file = directory.with_suffix(".zip")
    compression = ZIP_DEFLATED if compress else ZIP_STORED
    with ZipFile(zip_file, "w", compression) as f:
        for file in TQDM(files_to_zip, desc=f"正在压缩 {directory} 到 {zip_file}...", unit="file", disable=not progress):
            f.write(file, file.relative_to(directory))

    return zip_file  # 返回压缩后的 zip 文件路径


def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX"), exist_ok=False, progress=True):
    """
    解压 *.zip 文件到指定路径，排除包含排除列表中字符串的文件。

    如果压缩文件没有包含单个顶级目录，函数将创建一个与压缩文件同名（不带扩展名）的新目录来提取其内容。
    如果未提供路径，函数将使用压缩文件的父目录作为默认路径。

    参数：
        file (str | Path): 要解压的压缩文件路径。
        path (str, 可选): 解压文件的目标路径。如果为 None，默认为压缩文件所在的父目录。
        exclude (tuple, 可选): 要排除的文件名字符串元组。默认为 ('.DS_Store', '__MACOSX')。
        exist_ok (bool, 可选): 是否覆盖已存在的内容。如果为 False，且目标目录已存在，函数会跳过解压。默认为 False。
        progress (bool, 可选): 是否显示进度条。默认为 True。

    异常：
        BadZipFile: 如果提供的文件不存在或不是有效的压缩文件。

    返回：
        (Path): 解压后的目录路径。

    示例：
        ```python
        from ultralytics.utils.downloads import unzip_file

        dir = unzip_file("path/to/file.zip")
        ```
    """
    from zipfile import BadZipFile, ZipFile, is_zipfile

    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"文件 '{file}' 不存在或不是有效的压缩文件。")
    if path is None:
        path = Path(file).parent  # 默认路径

    # 解压文件内容
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in files}

        # 决定是否直接解压还是解压到一个目录中
        unzip_as_dir = len(top_level_dirs) == 1  # 判断是否有单个顶级目录
        if unzip_as_dir:
            # 压缩包包含 1 个顶级目录
            extract_path = path  # 即解压到 ../datasets
            path = Path(path) / list(top_level_dirs)[0]  # 即解压到 ../datasets/ 下的 coco8/ 目录
        else:
            # 压缩包包含多个顶级文件
            path = extract_path = Path(path) / Path(file).stem  # 即解压多个文件到 ../datasets/coco8/

        # 检查目标目录是否已存在且包含文件
        if path.exists() and any(path.iterdir()) and not exist_ok:
            # 如果目录存在且不为空，跳过解压，直接返回路径
            LOGGER.warning(f"警告 ⚠️ 跳过解压 {file}，因为目标目录 {path} 不为空。")
            return path

        for f in TQDM(files, desc=f"正在解压 {file} 到 {Path(path).resolve()}...", unit="file", disable=not progress):
            # 确保文件路径不包含 "上级目录" 来避免路径遍历安全漏洞
            if ".." in Path(f).parts:
                LOGGER.warning(f"潜在不安全的文件路径: {f}，跳过解压。")
                continue
            zipObj.extract(f, extract_path)

    return path  # 返回解压目录


def check_disk_space(url="https://ultralytics.com/assets/coco8.zip", path=Path.cwd(), sf=1.5, hard=True):
    """
    检查是否有足够的磁盘空间来下载并存储文件。

    参数：
        url (str, 可选): 文件的 URL。默认为 'https://ultralytics.com/assets/coco8.zip'。
        path (str | Path, 可选): 要检查可用空间的路径或磁盘。默认为当前工作目录。
        sf (float, 可选): 安全系数，是所需空间的倍数。默认为 1.5。
        hard (bool, 可选): 磁盘空间不足时是否抛出错误。默认为 True。

    返回：
        (bool): 如果磁盘空间足够，返回 True，否则返回 False。
    """
    try:
        r = requests.head(url)  # 响应
        assert r.status_code < 400, f"URL 错误 {url}: {r.status_code} {r.reason}"  # 检查响应
    except Exception:
        return True  # 请求问题，默认返回 True

    # 检查文件大小
    gib = 1 << 30  # 每 GiB 的字节数
    data = int(r.headers.get("Content-Length", 0)) / gib  # 文件大小（GB）
    total, used, free = (x / gib for x in shutil.disk_usage(path))  # 获取磁盘空间信息

    if data * sf < free:
        return True  # 足够空间

    # 空间不足
    text = (
        f"警告 ⚠️ 可用磁盘空间 {free:.1f} GB 小于 {data * sf:.3f} GB，"
        f"请释放 {data * sf - free:.1f} GB 额外磁盘空间后再试。"
    )
    if hard:
        raise MemoryError(text)
    LOGGER.warning(text)
    return False


def get_google_drive_file_info(link):
    """
    获取可分享的 Google Drive 文件链接的直接下载链接和文件名。

    参数：
        link (str): Google Drive 文件的可分享链接。

    返回：
        (str): Google Drive 文件的直接下载链接。
        (str): Google Drive 文件的原始文件名。如果提取失败，返回 None。

    示例：
        ```python
        from ultralytics.utils.downloads import get_google_drive_file_info

        link = "https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link"
        url, filename = get_google_drive_file_info(link)
        ```
    """
    file_id = link.split("/d/")[1].split("/view")[0]
    drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    filename = None

    # 启动会话
    with requests.Session() as session:
        response = session.get(drive_url, stream=True)
        if "quota exceeded" in str(response.content.lower()):
            raise ConnectionError(
                emojis(
                    f"❌  Google Drive 文件下载配额已满。"
                    f"请稍后再试或手动下载该文件，链接为 {link}。"
                )
            )
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                drive_url += f"&confirm={v}"  # v 是确认 token
        if cd := response.headers.get("content-disposition"):
            filename = re.findall('filename="(.+)"', cd)[0]
    return drive_url, filename


def safe_download(
    url,
    file=None,
    dir=None,
    unzip=True,
    delete=False,
    curl=False,
    retry=3,
    min_bytes=1e0,
    exist_ok=False,
    progress=True,
):
    """
    从URL下载文件，支持重试、解压和删除已下载文件的选项。

    参数:
        url (str): 要下载的文件的URL。
        file (str, 可选): 下载的文件名。如果未提供，将使用URL的文件名。
        dir (str, 可选): 要保存下载文件的目录。如果未提供，文件将保存在当前工作目录中。
        unzip (bool, 可选): 是否解压下载的文件。默认为True。
        delete (bool, 可选): 是否在解压后删除下载的文件。默认为False。
        curl (bool, 可选): 是否使用curl命令行工具进行下载。默认为False。
        retry (int, 可选): 下载失败时重试的次数。默认为3。
        min_bytes (float, 可选): 下载的文件必须具有的最小字节数，才能认为是成功的下载。默认为1E0。
        exist_ok (bool, 可选): 解压时是否覆盖现有内容。默认为False。
        progress (bool, 可选): 是否在下载时显示进度条。默认为True。

    示例:
        ```python
        from ultralytics.utils.downloads import safe_download

        link = "https://ultralytics.com/assets/bus.jpg"
        path = safe_download(link)
        ```
    """
    gdrive = url.startswith("https://drive.google.com/")  # 检查是否是Google Drive链接
    if gdrive:
        url, file = get_google_drive_file_info(url)

    f = Path(dir or ".") / (file or url2file(url))  # 将URL转换为文件名
    if "://" not in str(url) and Path(url).is_file():  # 检查URL是否存在 ('://' 检查在Windows Python <3.10中需要)
        f = Path(url)  # 文件名
    elif not f.is_file():  # URL和文件都不存在
        uri = (url if gdrive else clean_url(url)).replace(  # 清理和别名化的URL
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/",
            "https://ultralytics.com/assets/",  # 资产别名
        )
        desc = f"正在下载 {uri} 到 '{f}'"
        LOGGER.info(f"{desc}...")
        f.parent.mkdir(parents=True, exist_ok=True)  # 如果目录不存在，则创建目录
        check_disk_space(url, path=f.parent)
        for i in range(retry + 1):
            try:
                if curl or i > 0:  # 使用curl下载并重试
                    s = "sS" * (not progress)  # 静默模式
                    r = subprocess.run(["curl", "-#", f"-{s}L", url, "-o", f, "--retry", "3", "-C", "-"]).returncode
                    assert r == 0, f"Curl返回值 {r}"
                else:  # 使用urllib下载
                    method = "torch"
                    if method == "torch":
                        torch.hub.download_url_to_file(url, f, progress=progress)
                    else:
                        with request.urlopen(url) as response, TQDM(
                            total=int(response.getheader("Content-Length", 0)),
                            desc=desc,
                            disable=not progress,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as pbar:
                            with open(f, "wb") as f_opened:
                                for data in response:
                                    f_opened.write(data)
                                    pbar.update(len(data))

                if f.exists():
                    if f.stat().st_size > min_bytes:
                        break  # 下载成功
                    f.unlink()  # 删除部分下载的文件
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(emojis(f"❌  下载失败 {uri}，环境不可在线。")) from e
                elif i >= retry:
                    raise ConnectionError(emojis(f"❌  下载失败 {uri}，已达到重试次数限制。")) from e
                LOGGER.warning(f"⚠️ 下载失败，正在重试 {i + 1}/{retry} {uri}...")

    if unzip and f.exists() and f.suffix in {"", ".zip", ".tar", ".gz"}:
        from zipfile import is_zipfile

        unzip_dir = (dir or f.parent).resolve()  # 如果提供了目录，则解压到该目录，否则在原地解压
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress)  # 解压
        elif f.suffix in {".tar", ".gz"}:
            LOGGER.info(f"正在解压 {f} 到 {unzip_dir}...")
            subprocess.run(["tar", "xf" if f.suffix == ".tar" else "xfz", f, "--directory", unzip_dir], check=True)
        if delete:
            f.unlink()  # 删除zip文件
        return unzip_dir


def get_github_assets(repo="ultralytics/assets", version="latest", retry=False):
    """
    从GitHub仓库获取指定版本的标签和资产。如果未指定版本，函数将获取最新的发布版本的资产。

    参数:
        repo (str, 可选): GitHub仓库，格式为 'owner/repo'。默认为 'ultralytics/assets'。
        version (str, 可选): 要获取资产的发布版本。默认为 'latest'。
        retry (bool, 可选): 下载失败时是否重试。默认为False。

    返回:
        (tuple): 返回一个元组，包含发布标签和资产名称列表。

    示例:
        ```python
        tag, assets = get_github_assets(repo="ultralytics/assets", version="latest")
        ```
    """
    if version != "latest":
        version = f"tags/{version}"  # 即 tags/v6.2
    url = f"https://api.github.com/repos/{repo}/releases/{version}"
    r = requests.get(url)  # GitHub API请求
    if r.status_code != 200 and r.reason != "rate limit exceeded" and retry:  # 如果失败并且不是403限制
        r = requests.get(url)  # 重试
    if r.status_code != 200:
        LOGGER.warning(f"⚠️ GitHub资产检查失败 {url}: {r.status_code} {r.reason}")
        return "", []
    data = r.json()
    return data["tag_name"], [x["name"] for x in data["assets"]]  # 返回标签和资产名称列表，例如 ['yolov8n.pt', 'yolov8s.pt', ...]


def attempt_download_asset(file, repo="ultralytics/assets", release="v8.3.0", **kwargs):
    """
    如果本地未找到文件，尝试从 GitHub 发布资产中下载文件。该函数首先检查本地是否有文件，
    如果没有，再尝试从指定的 GitHub 仓库发布中下载文件。

    参数:
        file (str | Path): 要下载的文件名或文件路径。
        repo (str, 可选): GitHub 仓库，格式为 'owner/repo'。默认为 'ultralytics/assets'。
        release (str, 可选): 要下载的具体版本。默认为 'v8.3.0'。
        **kwargs (any): 下载过程中其他关键字参数。

    返回:
        (str): 下载文件的路径。

    示例:
        ```python
        file_path = attempt_download_asset("yolo11n.pt", repo="ultralytics/assets", release="latest")
        ```
    """
    from ultralytics.utils import SETTINGS  # 为避免循环导入，作用域限定

    if 'v12' in str(file):
        repo = "sunsmarterjie/yolov12"
        release = "turbo"

    # YOLOv3/5u 更新
    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ""))
    if file.exists():
        return str(file)
    elif (SETTINGS["weights_dir"] / file).exists():
        return str(SETTINGS["weights_dir"] / file)
    else:
        # URL 指定
        name = Path(parse.unquote(str(file))).name  # 解码 '%2F' 为 '/' 等
        download_url = f"https://github.com/{repo}/releases/download"
        if str(file).startswith(("http:/", "https:/")):  # 下载
            url = str(file).replace(":/", "://")  # Pathlib 会将 :// 转换为 :/
            file = url2file(name)  # 解析身份验证 https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f"在 {file} 本地找到 {clean_url(url)}")  # 文件已存在
            else:
                safe_download(url=url, file=file, min_bytes=1e5, **kwargs)

        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:
            safe_download(url=f"{download_url}/{release}/{name}", file=file, min_bytes=1e5, **kwargs)

        else:
            tag, assets = get_github_assets(repo, release)
            if not assets:
                tag, assets = get_github_assets(repo)  # 最新发布
            if name in assets:
                safe_download(url=f"{download_url}/{tag}/{name}", file=file, min_bytes=1e5, **kwargs)

        return str(file)


def download(url, dir=Path.cwd(), unzip=True, delete=False, curl=False, threads=1, retry=3, exist_ok=False):
    """
    从指定的 URL 下载文件到给定的目录。如果指定了多个线程，还支持并发下载。

    参数:
        url (str | list): 要下载的文件的 URL 或 URL 列表。
        dir (Path, 可选): 文件保存的目录。默认为当前工作目录。
        unzip (bool, 可选): 下载后是否解压文件。默认为 True。
        delete (bool, 可选): 解压后是否删除压缩文件。默认为 False。
        curl (bool, 可选): 是否使用 curl 下载。默认为 False。
        threads (int, 可选): 用于并发下载的线程数。默认为 1。
        retry (int, 可选): 下载失败时的重试次数。默认为 3。
        exist_ok (bool, 可选): 解压时是否覆盖已有内容。默认为 False。

    示例:
        ```python
        download("https://ultralytics.com/assets/example.zip", dir="path/to/dir", unzip=True)
        ```
    """
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # 创建目录
    if threads > 1:
        with ThreadPool(threads) as pool:
            pool.map(
                lambda x: safe_download(
                    url=x[0],
                    dir=x[1],
                    unzip=unzip,
                    delete=delete,
                    curl=curl,
                    retry=retry,
                    exist_ok=exist_ok,
                    progress=threads <= 1,
                ),
                zip(url, repeat(dir)),
            )
            pool.close()
            pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            safe_download(url=u, dir=dir, unzip=unzip, delete=delete, curl=curl, retry=retry, exist_ok=exist_ok)
