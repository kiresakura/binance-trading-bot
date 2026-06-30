#!/usr/bin/env python3
"""获取当前 IP 地址信息，用于 Binance API 白名单设置"""

import requests
import json

def get_ip_info():
    """获取详细的 IP 信息"""
    print("=" * 60)
    print("🌐 获取你的 IP 地址信息")
    print("=" * 60)
    print()
    
    # 获取 IPv4 地址
    try:
        ipv4 = requests.get('https://api.ipify.org?format=text', timeout=5).text.strip()
        print(f"📍 你的公网 IPv4 地址: {ipv4}")
    except Exception as e:
        print(f"❌ 无法获取 IPv4: {e}")
        ipv4 = None
    
    # 获取 IPv6 地址
    try:
        ipv6 = requests.get('https://api6.ipify.org?format=text', timeout=5).text.strip()
        print(f"📍 你的公网 IPv6 地址: {ipv6}")
    except Exception as e:
        print(f"ℹ️  IPv6: 未检测到")
        ipv6 = None
    
    print()
    
    # 获取详细地理信息
    if ipv4:
        try:
            print("🔍 获取 IP 详细信息...")
            response = requests.get(f'https://ipapi.co/{ipv4}/json/', timeout=5)
            data = response.json()
            
            print(f"   国家: {data.get('country_name', 'N/A')}")
            print(f"   地区: {data.get('region', 'N/A')}")
            print(f"   城市: {data.get('city', 'N/A')}")
            print(f"   ISP: {data.get('org', 'N/A')}")
        except Exception as e:
            print(f"   详细信息获取失败: {e}")
    
    print()
    print("=" * 60)
    print("📋 Binance API 白名单设置建议")
    print("=" * 60)
    print()
    
    if ipv4:
        print(f"✅ 推荐添加到 Binance API 白名单:")
        print(f"   {ipv4}")
        print()
        print("⚠️  注意事项:")
        print()
        print("1. 如果你的网络使用动态 IP（每次重启路由器会变）：")
        print("   - 家庭宽带通常是动态 IP")
        print("   - 建议不设置白名单，或使用其他安全措施")
        print()
        print("2. 如果你的网络使用静态 IP（IP 不会变）：")
        print("   - 企业网络、某些商业宽带")
        print("   - 强烈建议设置白名单")
        print()
        print("3. 如果你使用云服务器部署机器人：")
        print("   - 云服务器通常有固定公网 IP")
        print("   - 务必设置白名单！")
    
    print()
    print("=" * 60)
    print("🔧 如何在 Binance 设置白名单")
    print("=" * 60)
    print()
    print("【测试网】:")
    print("  - 测试网不支持 IP 白名单")
    print("  - 可以直接使用")
    print()
    print("【真实交易】:")
    print("  1. 登录 Binance.com")
    print("  2. 进入 账户 → API 管理")
    print("  3. 选择你的 API Key → 编辑")
    print("  4. 找到 'IP 访问限制'")
    print("  5. 选择 '限制访问受信任的 IP（推荐）'")
    if ipv4:
        print(f"  6. 输入: {ipv4}")
    print("  7. 保存更改")
    print()
    print("=" * 60)
    print("💡 测试 IP 是否会变化")
    print("=" * 60)
    print()
    print("运行此命令多次（重启路由器前后）查看 IP 是否变化:")
    print("  python3 get_my_ip.py")
    print()
    print("或使用:")
    print("  curl ifconfig.me")
    print()

if __name__ == '__main__':
    get_ip_info()
