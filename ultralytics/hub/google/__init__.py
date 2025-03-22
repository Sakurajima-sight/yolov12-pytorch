# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import concurrent.futures
import statistics
import time
from typing import List, Optional, Tuple

import requests


class GCPRegions:
    """
    一个用于管理和分析Google Cloud Platform（GCP）区域的类。

    该类提供了初始化、分类和分析GCP区域的功能，基于其地理位置、层级分类和网络延迟。

    属性：
        regions (Dict[str, Tuple[int, str, str]]): 一个包含GCP区域及其层级、城市和国家信息的字典。

    方法：
        tier1: 返回一个包含tier 1 GCP区域的列表。
        tier2: 返回一个包含tier 2 GCP区域的列表。
        lowest_latency: 确定具有最低网络延迟的GCP区域。

    示例：
        >>> from ultralytics.hub.google import GCPRegions
        >>> regions = GCPRegions()
        >>> lowest_latency_region = regions.lowest_latency(verbose=True, attempts=3)
        >>> print(f"最低延迟区域: {lowest_latency_region[0][0]}")
    """

    def __init__(self):
        """初始化GCPRegions类，并预定义了Google Cloud Platform区域及其相关信息。"""
        self.regions = {
            "asia-east1": (1, "台湾", "中国"),
            "asia-east2": (2, "香港", "中国"),
            "asia-northeast1": (1, "东京", "日本"),
            "asia-northeast2": (1, "大阪", "日本"),
            "asia-northeast3": (2, "首尔", "韩国"),
            "asia-south1": (2, "孟买", "印度"),
            "asia-south2": (2, "德里", "印度"),
            "asia-southeast1": (2, "裕廊西", "新加坡"),
            "asia-southeast2": (2, "雅加达", "印度尼西亚"),
            "australia-southeast1": (2, "悉尼", "澳大利亚"),
            "australia-southeast2": (2, "墨尔本", "澳大利亚"),
            "europe-central2": (2, "华沙", "波兰"),
            "europe-north1": (1, "哈米纳", "芬兰"),
            "europe-southwest1": (1, "马德里", "西班牙"),
            "europe-west1": (1, "圣吉斯兰", "比利时"),
            "europe-west10": (2, "柏林", "德国"),
            "europe-west12": (2, "都灵", "意大利"),
            "europe-west2": (2, "伦敦", "英国"),
            "europe-west3": (2, "法兰克福", "德国"),
            "europe-west4": (1, "伊姆沙芬", "荷兰"),
            "europe-west6": (2, "苏黎世", "瑞士"),
            "europe-west8": (1, "米兰", "意大利"),
            "europe-west9": (1, "巴黎", "法国"),
            "me-central1": (2, "多哈", "卡塔尔"),
            "me-west1": (1, "特拉维夫", "以色列"),
            "northamerica-northeast1": (2, "蒙特利尔", "加拿大"),
            "northamerica-northeast2": (2, "多伦多", "加拿大"),
            "southamerica-east1": (2, "圣保罗", "巴西"),
            "southamerica-west1": (2, "圣地亚哥", "智利"),
            "us-central1": (1, "爱荷华州", "美国"),
            "us-east1": (1, "南卡罗来纳州", "美国"),
            "us-east4": (1, "北弗吉尼亚", "美国"),
            "us-east5": (1, "哥伦布", "美国"),
            "us-south1": (1, "达拉斯", "美国"),
            "us-west1": (1, "俄勒冈州", "美国"),
            "us-west2": (2, "洛杉矶", "美国"),
            "us-west3": (2, "盐湖城", "美国"),
            "us-west4": (2, "拉斯维加斯", "美国"),
        }

    def tier1(self) -> List[str]:
        """返回所有GCP区域中被分类为tier 1的区域列表。"""
        return [region for region, info in self.regions.items() if info[0] == 1]

    def tier2(self) -> List[str]:
        """返回所有GCP区域中被分类为tier 2的区域列表。"""
        return [region for region, info in self.regions.items() if info[0] == 2]

    @staticmethod
    def _ping_region(region: str, attempts: int = 1) -> Tuple[str, float, float, float, float]:
        """对指定的GCP区域进行ping测试，并返回延迟统计信息：均值、最小值、最大值和标准差。"""
        url = f"https://{region}-docker.pkg.dev"
        latencies = []
        for _ in range(attempts):
            try:
                start_time = time.time()
                _ = requests.head(url, timeout=5)
                latency = (time.time() - start_time) * 1000  # 将延迟转换为毫秒
                if latency != float("inf"):
                    latencies.append(latency)
            except requests.RequestException:
                pass
        if not latencies:
            return region, float("inf"), float("inf"), float("inf"), float("inf")

        std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
        return region, statistics.mean(latencies), std_dev, min(latencies), max(latencies)

    def lowest_latency(
        self,
        top: int = 1,
        verbose: bool = False,
        tier: Optional[int] = None,
        attempts: int = 1,
    ) -> List[Tuple[str, float, float, float, float]]:
        """
        根据ping测试确定具有最低延迟的GCP区域。

        参数：
            top (int): 返回的最 top 区域数量。
            verbose (bool): 如果为True，将打印所有测试区域的详细延迟信息。
            tier (int | None): 按层级筛选区域（1或2）。如果为None，将测试所有区域。
            attempts (int): 每个区域的ping测试次数。

        返回：
            (List[Tuple[str, float, float, float, float]]): 返回包含区域信息和延迟统计数据的元组列表。
            每个元组包含 (区域, 平均延迟, 标准差, 最小延迟, 最大延迟)。

        示例：
            >>> regions = GCPRegions()
            >>> results = regions.lowest_latency(top=3, verbose=True, tier=1, attempts=2)
            >>> print(results[0][0])  # 打印最低延迟区域的名称
        """
        if verbose:
            print(f"正在测试GCP区域的延迟（尝试次数：{attempts}）...")

        regions_to_test = [k for k, v in self.regions.items() if v[0] == tier] if tier else list(self.regions.keys())
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            results = list(executor.map(lambda r: self._ping_region(r, attempts), regions_to_test))

        sorted_results = sorted(results, key=lambda x: x[1])

        if verbose:
            print(f"{'区域':<25} {'位置':<35} {'层级':<5} 延迟（毫秒）")
            for region, mean, std, min_, max_ in sorted_results:
                tier, city, country = self.regions[region]
                location = f"{city}, {country}"
                if mean == float("inf"):
                    print(f"{region:<25} {location:<35} {tier:<5} 超时")
                else:
                    print(f"{region:<25} {location:<35} {tier:<5} {mean:.0f} ± {std:.0f} ({min_:.0f} - {max_:.0f})")
            print(f"\n最低延迟区域{'s' if top > 1 else ''}:")
            for region, mean, std, min_, max_ in sorted_results[:top]:
                tier, city, country = self.regions[region]
                location = f"{city}, {country}"
                print(f"{region} ({location}, {mean:.0f} ± {std:.0f} 毫秒 ({min_:.0f} - {max_:.0f}))")

        return sorted_results[:top]


# 使用示例
if __name__ == "__main__":
    regions = GCPRegions()
    top_3_latency_tier1 = regions.lowest_latency(top=3, verbose=True, tier=1, attempts=3)
