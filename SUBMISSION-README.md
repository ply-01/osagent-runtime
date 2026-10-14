# 开放原子 2026 赛道一源码交付

通用安装、演示与验证入口见 README.md。本快照含 Python 后端、C++ 桥接源码、
服务接入、公开仿真样例、部署脚本和测试。它不是离线依赖安装器。
原始来源声明 commit：6ca20ca8155ae148cf95d874e068cf287e00cadf；当前改造快照没有 Git 历史，
该 commit 不代表新代码的版本。新的交付清单由 scripts/package-track1.py 生成。

## 可选 Open WebUI

源码包不包含 Open WebUI 全仓；离线演示不依赖该组件。
需要此前部署前端时可获取上游固定版本：

```bash
git clone https://github.com/open-webui/open-webui.git vendor/open-webui
git -C vendor/open-webui checkout f9590b8017199e56d5e953657e6498e3cef1d246
git -C vendor/open-webui apply ../patches/open-webui/0001-add-osagent-plugin-navigation.patch
```

保留上游许可证、通知和品牌要求，核对固定版本条款。
当前快照仅包含 0001 补丁，不宣称交付旧文档中提到的额外补丁。

## 交付边界

Python 依赖按 pyproject.toml 安装；uv.lock 可供选择相同版本。
系统 SDK、模型、私有数据集、API Key、用户记忆和原始运行缓存不在源码包中。
麒麟系统组件及授权由目标机器提供，不能按项目 Apache-2.0 重新授权。
公开场景报告放在独立 artifacts 目录，可单独附上，默认源码包不包含运行状态。
正式 PDF、演示视频及团队联系信息位于独立的本地提交材料目录，不混入公开源码。
仓库和视频发布地址仍需补充，原代码权属须确认；详见 doc/track1/submission.md。

## 材料重建和检查

先运行公开场景及测试，保存实际证据至 artifacts/submission-evidence。
scripts/build-track1-materials.py 读取通过 --info 指定的本地团队信息，生成 PDF 和字幕视频。
文档依赖 reportlab、Pillow；视频可用 imageio-ffmpeg，macOS 也可使用系统 Swift / AVFoundation。
这些仅用于材料生成，不是运行平台核心依赖。中文字体由操作者合法安装，不打包字体文件。
scripts/verify-track1-video.swift 可在 macOS 完整解码核验视频。

```bash
python scripts/check-submission.py --directory /path/to/submission --json
python scripts/check-submission.py --directory /path/to/submission --require-ready
```

第二条还要求填写仓库、视频 HTTPS 地址并确认公开授权。地址语法检查不能代替匿名访问核验。
