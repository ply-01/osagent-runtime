#!/usr/bin/env python3
"""Audit local submission materials separately from externally required links."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile


def audit(directory: Path) -> dict:
    checks=[];blockers=[]
    def check(name,passed,detail):checks.append({'name':name,'passed':bool(passed),'detail':detail})
    try:
        info=json.loads((directory/'project-info.json').read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:
        return {'local_complete':False,'submission_ready':False,'checks':[], 'blockers':[f'提交信息不可读：{exc}']}
    check('项目与主体信息',all(info.get(k) for k in ['project_name','track','organization','leader','email','phone']),'核对实际报名资料；联系方式不进入源码包')
    members=info.get('members',[])
    check('团队规模和负责人',isinstance(members,list) and 1<=len(members)<=8 and len(set(members))==len(members) and info.get('leader') in members,'1 至 8 人，负责人在成员名单内')
    pdf=directory/'01-作品介绍.pdf'
    check('作品介绍 PDF',pdf.is_file() and pdf.read_bytes().startswith(b'%PDF-'),'正式标题、背景、架构、应用场景、创新点和验证边界')
    video=directory/'02-核心功能演示.mp4'
    head=video.read_bytes()[:64] if video.is_file() else b''
    check('演示视频 MP4',b'ftyp' in head and video.stat().st_size>1024 if video.is_file() else False,'实际运行结果可视化演示；公开合成输入')
    archive=directory/'03-完整源码.zip'
    try:
        with ZipFile(archive) as bundle:
            manifest=json.loads(bundle.read('osagent/SOURCE-MANIFEST.json'))
            exact=all(hashlib.sha256(bundle.read('osagent/'+name)).hexdigest()==digest for name,digest in manifest['sha256'].items())
            names=bundle.namelist()
            excluded=all(not any(p in name.split('/') for p in ['.venv','__pycache__','data','memory']) or name.startswith('osagent/src/os_agent/memory/') for name in names)
            excluded=excluded and not any(name.endswith('.local.json') or Path(name).name.startswith('.env') for name in names)
            check('源码包哈希和 CRC',exact and bundle.testzip() is None,f'{len(manifest["sha256"])} 个文件逐字节 SHA-256')
            check('源码包排除运行与私有数据',excluded,'不含虚拟环境、用户记忆、私有配置或团队联系方式')
            check('许可与治理文件',all('osagent/'+name in names for name in ['LICENSE','THIRD_PARTY_NOTICES.md','CONTRIBUTING.md','GOVERNANCE.md','ROADMAP.md','SECURITY.md']),'自有许可方案与第三方声明；公开授权另行确认')
            check('提交信息未混入源码包',not any('project-info.json' in name for name in names),'团队联系方式只存在提交材料中')
    except (OSError,ValueError,KeyError) as exc:
        check('源码包校验',False,str(exc))
    try:
        report=json.loads((directory/'04-验证证据/scenario.json').read_text(encoding='utf-8'))
        check('公开场景运行证据',report.get('passed') and report.get('checks_total',0)>0 and report['checks_passed']==report['checks_total'],'与视频、HTML 报告使用同一次真实运行结果')
    except (OSError,ValueError,KeyError) as exc:check('公开场景运行证据',False,str(exc))
    for key,label in [('repository_url','代码仓库链接'),('video_url','演示视频链接')]:
        parsed=urlparse(info.get(key,''))
        if parsed.scheme!='https' or not parsed.netloc or 'example.' in parsed.netloc:
            blockers.append(label+'尚未提供有效的 HTTPS 发布地址')
    if not info.get('code_rights_confirmed'):
        blockers.append('原代码权属与 Apache-2.0 公开授权尚未明确确认')
    local=all(c['passed'] for c in checks)
    if not local:blockers.append('本地材料检查存在失败项')
    return {'local_complete':local,'submission_ready':local and not blockers,'checks':checks,'blockers':blockers,
            'note':'HTTPS 地址语法通过不等于评审可访问；发布后仍需匿名访问核验。报名确认邮件补充要求需人工核对。'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--json',action='store_true');parser.add_argument('--require-ready',action='store_true');args=parser.parse_args()
    report=audit(args.directory)
    if args.json:print(json.dumps(report,ensure_ascii=False,indent=2))
    else:
        print('本地材料：'+('通过' if report['local_complete'] else '未通过'))
        print('可正式提交：'+('是' if report['submission_ready'] else '否'))
        for item in report['checks']:print(('PASS ' if item['passed'] else 'FAIL ')+item['name']+'：'+item['detail'])
        for blocker in report['blockers']:print('待完成：'+blocker)
    raise SystemExit(0 if (report['submission_ready'] if args.require_ready else report['local_complete']) else 1)
