#!/usr/bin/env python3
"""Build the submission PDF and subtitled video from actual public run evidence.

Artifact-only dependencies: reportlab and Pillow, plus imageio-ffmpeg or the
macOS Swift / AVFoundation encoder. They are not runtime requirements.
Set OSAGENT_CJK_FONT to a legally installed CJK TTF.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = Path(os.environ.get('OSAGENT_CJK_FONT', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))


def build_pdf(output, info, report, tests):
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    pdfmetrics.registerFont(TTFont('CJK', str(FONT_PATH)))
    W,H=A4; navy=HexColor('#132b46'); teal=HexColor('#087f85'); muted=HexColor('#546478')
    c=canvas.Canvas(str(output), pagesize=A4)
    c.setTitle(info['project_name']);c.setAuthor(info['organization']+' / '+info['leader'])
    style=ParagraphStyle('body',fontName='CJK',fontSize=10.5,leading=18,textColor=navy,wordWrap='CJK')
    small=ParagraphStyle('small',parent=style,fontSize=9,leading=14)
    heading=ParagraphStyle('heading',parent=style,fontSize=14,leading=21,textColor=teal)
    y=0;page=0
    def paragraph(text,kind=style,gap=11):
        nonlocal y
        p=Paragraph(text,kind);pw,ph=p.wrap(W-96,y-65)
        if y-ph<65: raise ValueError(f'page {page} overflow: {text[:60]}')
        p.drawOn(c,48,y-ph);y-=ph+gap
    def begin(title,kicker):
        nonlocal y,page
        page+=1;c.setFillColor(navy);c.rect(0,H-76,W,76,fill=1,stroke=0)
        c.setFont('CJK',9);c.setFillColor(HexColor('#b4dce2'));c.drawString(48,H-30,kicker)
        c.setFillColor(white);c.setFont('CJK',19);c.drawString(48,H-59,title);y=H-106
    def finish():
        c.setStrokeColor(HexColor('#dbe3ea'));c.line(48,48,W-48,48)
        c.setFont('CJK',8);c.setFillColor(muted);c.drawString(48,32,'OSAgent Runtime | 开放原子 2026 · 赛道一')
        c.drawRightString(W-48,32,f'{page} / 7');c.showPage()
    def section(t):paragraph(t,heading,7)
    def table(rows,widths):
        nonlocal y
        data=[[Paragraph(escape(str(v)),small) for v in row] for row in rows]
        t=Table(data,colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#e7f3f4')),('GRID',(0,0),(-1,-1),.4,HexColor('#d6e1e8')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
        tw,th=t.wrap(W-96,y-65)
        if y-th<65:raise ValueError(f'table overflow page {page}')
        t.drawOn(c,48,y-th);y-=th+14
    passed,skipped=tests
    # Cover
    begin('作品介绍', 'OPEN SOURCE INDUSTRY SOLUTION / 2026')
    paragraph('OSAgent Runtime',ParagraphStyle('hero',parent=style,fontSize=31,leading=43),15)
    paragraph('操作系统智能体开发运行平台',ParagraphStyle('subtitle',parent=style,fontSize=21,leading=31),18)
    paragraph('有界任务执行 · 可审计长期记忆 · MCP / HTTP 接入',heading,22)
    paragraph('面向开发者和行业应用构建者，统一任务规划、工具调用、事件轨迹与长期记忆。以制造运维知识交接的公开仿真，展示平台核心记忆能力的可部署、可运行、可测试闭环。')
    table([['参赛信息','内容'],['参赛赛道',info['track']],['申报单位',info['organization']+'（高校）'],['负责人',info['leader']],['团队成员','、'.join(info['members'])],['交付形式','Python 软件平台、源码、作品说明、运行证据与演示视频']],[100,W-196])
    section('本次已经验证')
    paragraph(f'完整回归：{passed} 项通过，{skipped} 项可选前端检查跳过。公开合成场景：{report["checks_passed"]}/{report["checks_total"]} 项检查通过。新环境解压安装可复现。')
    paragraph('本材料使用公开合成输入和真实代码运行结果，不声明真实制造客户试点，不将仿真通过率等同自然语言准确率或比赛官方得分。',small)
    finish()
    # 2
    begin('项目背景与行业应用','01 / PROBLEM & APPLICATION')
    section('开发和运行智能体的共性问题')
    paragraph('任务执行需要统一规划、工具参数、权限边界、错误恢复和运行轨迹。跨会话工作还需要保存业务知识、当前约束及历史证据，否则应用容易重复询问、重复试错或使用已失效的信息。')
    paragraph('OSAgent Runtime 把运行时与记忆服务作为可复用的软件能力交付。应用可以通过 CLI、MCP 或 HTTP 接入，在保持本地记忆边界的同时替换模型与交互前端。')
    section('代表性场景：制造运维知识交接')
    table([['需求','平台提供的对应能力'],['跨班次交接','重新打开持久化存储后召回此前登记的维护流程'],['流程发生变更','更新当前事实，并保留被替代版本及审核来源'],['多个设备并存','以设备实体和作用域区分维护知识，检索显式限定对象'],['设备注销','通过记忆接口物理删除该设备的当前事实及旧版本']],[105,W-201])
    section('适用对象与落地方式')
    paragraph('适合构建桌面知识助手、运维辅助智能体与长周期业务工具的开发团队。可先在本机或隔离环境验证，然后在授权行业数据和明确权限策略下进行试点。制造样例用于展示平台能力，并非完整工厂控制系统。')
    section('需求与效果的证据边界')
    paragraph('本次有可复现的仿真工作流和软件测试证据。真实客户访谈、生产部署、成本收益及独立第三方测评尚未提供，后续应通过授权试点补充。')
    finish()
    # 3
    begin('技术架构与开源路线','02 / ARCHITECTURE')
    layers=[('应用接入','CLI / MCP Host / HTTP / 可选 Open WebUI'),('Agent 运行时','ContextBuilder / Planner / PlanValidator / AgentLoop / Hooks'),('工具与可观测性','文件与 PDF / 受控 Shell / 记忆接口 / Event / Session JSONL'),('统一记忆接口','remember / update / recall / forget'),('存储与检索','Core / Archival / Recall / STM；结构过滤、词法、可选语义与 RRF'),('平台扩展','macOS / Linux / Windows；本地文件与 SQLite')]
    for label,body in layers:
        c.setFillColor(HexColor('#eef4f8'));c.roundRect(48,y-60,W-96,60,7,fill=1,stroke=0)
        c.setFont('CJK',12);c.setFillColor(teal);c.drawString(62,y-22,label)
        c.setFont('CJK',9);c.setFillColor(navy);c.drawString(62,y-43,body)
        y-=75
    paragraph('核心运行链路使用三平台通用 Python 实现。语义模型与 Open WebUI 按需接入，分别配置、分别验收。',style)
    paragraph('源码定位：src/os_agent/agent/、tools/、events/、memory/；接入位于 mcp_server.py、web_service.py；行业样例入口为 track1.py。',small)
    finish()
    # 4
    begin('技术特点与创新边界','03 / TECHNICAL VALUE')
    section('1. 有边界、可观测的任务运行闭环')
    paragraph('AgentLoop 连接上下文构建、计划校验、工具执行、Hook 与事件日志。SimplePlanner 提供无模型的验证路径；LLMPlanner 允许接入兼容 API，但本次离线演示不验证模型推理质量。')
    section('2. 从片段检索到带证据的状态化记忆')
    paragraph('事实具有实体、属性、值、作用域、证据、来源会话和版本关系。Core 保存当前事实，Archival 保存被替代版本。更新与遗忘通过统一接口执行，便于追溯当前状态的依据。')
    section('3. 接入、记忆和模型解耦')
    paragraph('MCP 与 HTTP 复用后端，长期记忆不绑定某个前端。结构过滤、多路检索与上下文压缩用于控制证据范围和上下文体积。可选系统桥接提供国产桌面环境适配。')
    table([['能力','实现定位','本次证据'],['任务执行与轨迹','agent/loop.py、events/','规则规划、文件读取、Session 事件'],['事实更新与追溯','memory/module.py、mem0_core.py','新流程生效、旧流程进入历史'],['可复现的行业验证','track1.py、examples/track1/','JSON 输入、逐步真实输出与断言'],['三平台交付','embedding_provider.py、CI','通用存储与模型接口；远程 CI 分平台验收']],[106,164,W-366])
    paragraph('RRF、MCP、Open WebUI 与第三方模型不是本项目原创。本项目强调运行时与记忆状态机制的工程组织价值；没有新的独立同协议对比数据，不宣称算法首创或全面领先。',small)
    finish()
    # 5
    begin('核心功能演示与运行证据','04 / REPRODUCIBLE DEMO')
    paragraph('输入为公开合成的工厂、设备与审核单。演示直接调用真实记忆模块，使用词法检索，不加载外部模型。视频将任务运行时证据与以下记忆生命周期按顺序展示。')
    table([['操作','实际输出或检查'],['登记与跨实例持久化','工厂 A 的维护流程为 SOP-A1；重开存储后仍可召回'],['更新当前流程','工厂 A 新流程 SOP-A2 生效'],['查询被替代版本','历史中保留 SOP-A1 与原始登记证据'],['区分另一设备','工厂 B 仍召回 SOP-B1，不返回工厂 A 新流程'],['设备注销与物理遗忘','删除工厂 A 当前事实和旧版本；重开存储后无召回'],['全流程检查',f'{report["checks_passed"]}/{report["checks_total"]}；每次使用独立 run-* 状态目录']],[132,W-228])
    section('软件验证结果')
    paragraph(f'当前 macOS arm64 / Python 3.13.13：{passed} 项测试通过，{skipped} 项跳过，另有 30 个子测试通过。跳过项仅涉及未交付的可选 Open WebUI 源码/覆盖层检查，不代表前端验证通过。')
    paragraph('已验证源码包解压后在新虚拟环境安装、测试与运行场景。CLI、MCP 与 HTTP 的帮助入口可执行；具体核心服务行为由回归测试覆盖。跨平台 CI 配置已提供，尚未在 Windows/Linux 实机或远程 CI 运行。')
    paragraph('报告耗时只是本机小样例单次操作时间，不作为吞吐能力、10k 检索性能或生产 SLA。原始历史评测不与本次结果混算。',small)
    finish()
    # 6
    begin('开源治理与持续发展','05 / GOVERNANCE')
    section('许可和第三方边界')
    paragraph('自有代码采用 Apache-2.0 发布方案。原快照没有作者授权清单，公开发布前仍须负责人核实权属和贡献授权。Python 依赖按锁文件和包元数据保留许可证；模型与系统 SDK 遵守各自条款。')
    paragraph('可选 Open WebUI 使用固定上游版本及其独立许可证，保留版权、许可和品牌条件；项目许可证不能替代上游要求。当前核心包不包含上游全仓，行业演示不依赖该前端。')
    table([['治理材料','执行方式'],['贡献与评审','CONTRIBUTING、Issue/PR 模板；说明复现、验证、接口与依赖'],['持续集成','三系统、两 Python 版本配置；测试、仿真与源码包校验'],['发布与追溯','CHANGELOG、逐文件 SHA-256、源码包 CRC 检查'],['安全与数据','默认回环接入、有界工具策略；不提交密钥、私有配置或用户记忆']],[115,W-211])
    section('团队维护与发展规划')
    other_members = '、'.join(escape(name) for name in info['members'] if name != info['leader'])
    paragraph(f'团队规模 {len(info["members"])} 人，负责人{escape(info["leader"])}，成员{other_members}。维护职责和实际资源由团队内部落实；本材料不虚构导师、组织赞助、客户合作或长期资金支持。')
    paragraph('下一阶段：基于授权数据验证真实需求与效益；完善自然语言冲突、并发写入、备份恢复和身份隔离；按真实资源制定版本计划。新项目以可执行治理规划起步，不宣称已有成熟社区运营规模。')
    finish()
    # 7
    begin('部署使用与评审材料映射','06 / DELIVERY')
    section('最小复现路径')
    paragraph('Python 3.11+；首次安装依赖需要网络。完整回归还需 ripgrep。使用 uv 时可锁定安装：',style,4)
    paragraph('uv sync --frozen --extra dev<br/>uv run os-agent track1 --scenario \\<br/>examples/track1/maintenance-handover.json<br/>uv run python -m pytest -q',small)
    paragraph('或按 README 使用 venv + pip install -e ".[dev]"。打开本次 run-* 中的 report.html 查看输入、输出和检查结果。MCP 与 HTTP 接入见命令帮助和部署指南。')
    table([['官方评审维度','对应材料'],['技术创新 30%','运行时、状态化记忆、接口解耦与自主实现边界'],['场景落地 30%','公开制造运维仿真、安装运行、真实输出和回归证据'],['开源治理 20%','许可证与依赖、贡献流程、Issue/PR、CI 和哈希清单'],['长期发展 20%','负责人及团队、治理流程、风险与后续迭代计划']],[140,W-236])
    section('提交材料与限制')
    paragraph('本地交付包括作品介绍 PDF、核心功能演示 MP4、源码包和验证证据。官网同时要求可访问的代码仓库链接与演示视频链接；本地 ZIP 或 MP4 不能替代发布链接。源码已上传私有仓库，视频发布地址仍待补充。')
    paragraph('提交截止：2026 年 10 月 31 日 24:00；收件地址：bjoscc@oschina.cn。入围后 11 月 13 日线下路演。报名确认邮件中的补充要求以组委会通知为准。',small)
    paragraph('规则来源：www.oschina.net/oa2026/；按用户提供的官网规则核对。版本说明：2026-10-19 通用平台交付，仓库已创建，未自动报名或发送邮件。',small)
    finish();c.save()


def wrap_text(draw,text,font,width):
    lines=[];current=''
    for char in str(text):
        if char=='\n':lines.append(current);current='';continue
        if draw.textlength(current+char,font=font)>width and current:lines.append(current);current=char
        else:current+=char
    if current:lines.append(current)
    return lines


def build_video(output,report,runtime,tests,frames_dir):
    from PIL import Image,ImageDraw,ImageFont
    frames_dir.mkdir(parents=True,exist_ok=True)
    fonts={size:ImageFont.truetype(str(FONT_PATH),size) for size in [18,20,22,26,32,40,48]}
    chapters=[]
    def add(title,subtitle,input_text,output_text,duration=10):chapters.append(dict(title=title,subtitle=subtitle,input=input_text,output=output_text,duration=duration))
    add('OSAgent Runtime','操作系统智能体开发运行平台 | 北京航空航天大学','核心能力：有界任务运行、工具接入、事件轨迹与长期记忆。','本视频按实际运行日志可视化展示。制造运维数据为公开合成输入。',12)
    add('01 安装与启动','通用软件入口；支持 macOS / Linux / Windows；演示无需 API Key','uv sync --frozen --extra dev\nuv run os-agent run "列出文件并读取 README.md" --no-llm','下面展示规则规划路径执行真实文件工具，以及生成 Session 轨迹。',12)
    add('02 Agent 任务运行','真实 CLI 输出节选；规则规划，不是大模型推理','os-agent run "列出文件并读取 README.md"\n--no-llm --workspace . --session-id track1-runtime',runtime[runtime.find('OSAgent Runtime 已完成'):runtime.find('新增记忆:')].strip(),14)
    add('03 启动行业场景','制造运维知识交接 / 显式结构化输入','os-agent track1 --scenario\nexamples/track1/maintenance-handover.json','创建独立 run-* 目录，通过真实 AgentMemoryModule 执行持久化、更新、追溯与遗忘。',10)
    for index,step in enumerate(report['steps'],1):
        args=step['arguments'];result=step['result'];records=result.get('records',[])
        input_text=json.dumps(args,ensure_ascii=False,indent=2) if args else '重新打开同一磁盘状态目录。'
        if records:
            parts=[]
            for record in records[:2]:
                metadata=record.get('metadata',{})
                parts.append('\n'.join(f'{key}: {record[key]}' for key in ['entity','attribute','value','version','evidence'] if key in record))
                if metadata.get('version'):parts.append('version: '+str(metadata['version']))
                if metadata.get('evidence'):parts.append('evidence: '+str(metadata['evidence']))
                if metadata.get('hard_deleted'):parts.append('hard_deleted: true; purged_versions: '+str(metadata.get('purged_versions')))
            output_text='\n\n'.join(parts)
        else:output_text=result.get('message','')+'\nrecords: []'
        checks=step['checks'];passed=sum(c['passed'] for c in checks)
        status=f'检查 {passed}/{len(checks)} 通过' if checks else '操作成功'
        if not step['passed']:status='检查失败'
        add(f'{index+3:02d} {step["title"]}',step['action']+' | '+status,input_text,output_text,9)
    add('16 验证与开源交付','本机回归、独立安装及源码完整性',f'python -m pytest -q\n{tests[0]} passed / {tests[1]} skipped\n另有 30 个 subtests passed',f'公开场景检查 {report["checks_passed"]}/{report["checks_total"]} 通过。\n{tests[1]} 项跳过涉及缺失的可选前端源码，不计为通过。\n交付源码、许可证、治理文档及 SHA-256 清单。',12)
    add('17 使用范围与后续工作','仿真功能证据；不虚构生产客户、语义质量或自主桌面闭环','本次：macOS / Python 3.13.13\n无 LLM、无 Embedding 模型','后续：真实行业需求验证、授权数据试点、并发与恢复、身份隔离、独立质量对比。\n仓库公开访问、视频链接和代码权属确认仍需负责人完成。',12)
    concat=[];srt=[];cursor=0
    def clock(sec):
        ms=round(sec*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
    for index,ch in enumerate(chapters):
        image=Image.new('RGB',(1280,720),'#f1f5f8');d=ImageDraw.Draw(image)
        d.rectangle((0,0,1280,118),fill='#142c46')
        d.text((48,22),ch['title'],font=fonts[32],fill='white')
        d.text((48,73),ch['subtitle'],font=fonts[20],fill='#b8dce2')
        for x,label,text in [(48,'输入 / 操作',ch['input']),(656,'真实运行结果 / 证据',ch['output'])]:
            d.rounded_rectangle((x,145,x+576,589),radius=14,fill='white')
            d.text((x+22,163),label,font=fonts[26],fill='#087f85')
            lines=wrap_text(d,text,fonts[22],528)
            # Actual long outputs are excerpted; full evidence accompanies the video.
            if len(lines)>10:lines=lines[:9]+['… 完整输入与输出见随附 JSON / HTML 报告']
            for line_index,line in enumerate(lines):d.text((x+22,211+line_index*33),line,font=fonts[22],fill='#23364a')
        d.text((48,615),'公开合成场景 · 真实代码运行结果可视化 · 无旁白字幕演示',font=fonts[20],fill='#536579')
        d.text((48,652),f'{index+1:02d} / {len(chapters):02d}   |   OSAgent Runtime   |   开放原子 2026 · 赛道一',font=fonts[18],fill='#536579')
        d.rectangle((48,691,1232,698),fill='#d8e3eb');d.rectangle((48,691,48+int(1184*(index+1)/len(chapters)),698),fill='#087f85')
        frame=frames_dir/f'frame-{index:02}.png';image.save(frame)
        concat += ["file '"+frame.resolve().as_posix()+"'",f'duration {ch["duration"]}']
        srt.append(f'{index+1}\n{clock(cursor)} --> {clock(cursor+ch["duration"])}\n{ch["title"]}\n{ch["subtitle"]}\n')
        cursor+=ch['duration']
    concat.append("file '"+frame.resolve().as_posix()+"'")
    list_path=frames_dir/'frames.txt';list_path.write_text('\n'.join(concat)+'\n',encoding='utf-8')
    output.with_suffix('.srt').write_text('\n'.join(srt),encoding='utf-8')
    output.with_suffix('.chapters.json').write_text(json.dumps({'duration_seconds':cursor,'chapters':chapters},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    try:
        import imageio_ffmpeg
    except ImportError:
        import shutil
        if not shutil.which('swift'):
            raise RuntimeError('Video encoding needs imageio-ffmpeg or macOS Swift/AVFoundation')
        subprocess.run(['swift', str(ROOT/'scripts/encode-track1-video.swift'), str(frames_dir), str(output.with_suffix('.chapters.json')), str(output)], check=True)
        return cursor
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','concat','-safe','0','-i',str(list_path),'-r','30','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(output)]
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=open(frames_dir/'encode.log','w'))
    return cursor


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--info',type=Path,required=True)
    p.add_argument('--video',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    info=json.loads(a.info.read_text(encoding='utf-8'))
    report=json.loads((ROOT/'artifacts/submission-evidence/scenario.json').read_text(encoding='utf-8'))
    testlog=(ROOT/'artifacts/final-tests.log').read_text(encoding='utf-8');m=re.search(r'(\d+) passed, (\d+) skipped',testlog);tests=tuple(map(int,m.groups()))
    build_pdf(a.output/'01-作品介绍.pdf',info,report,tests)
    if a.video:
        runtime=(ROOT/'artifacts/submission-evidence/runtime-cli.txt').read_text(encoding='utf-8')
        duration=build_video(a.output/'02-核心功能演示.mp4',report,runtime,tests,ROOT/'artifacts/material-frames')
        print('Video duration:',duration)
    print(a.output)
