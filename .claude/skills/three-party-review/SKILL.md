---
name: three-party-review
allowed-tools: Bash, Read, Write, Edit
description: 多 agent 协作方法（三方公审）——一个实现者 + 两个独立评审者（实现审 / 设计审）并行把关每一步改动，三方都签才合主线。角色不绑 agent，claude / kimi / grok / codex 任一都能当任一角色，各自的无头起法在此。适用于任何需要多个 AI agent 分工改代码、写方案、互审的任务；用户说"三方公审""起 X 来审""多 agent 分工""写简报"时用。
---

## 方法一句话

改动切成**可单独评审的步**（一步 = 一个 commit 或一份方案）；每步由**实现者**做完并自核，然后**两个评审者并行起**：一个审实现（逐分支等价、实测对比、边界），一个审设计（与已签的方案是否一致、实现中新引入的取舍请裁、目标进度）；三方都「签」且必改已落 → 合主线，不再问人。任一不签 → 实现者修 → **只让不签的那方复审**。等评审时不阻塞：起完就汇报，定时器回看。

## 角色

| 角色 | 做什么 | 产出 |
|---|---|---|
| 实现者 | 改代码 / 写方案；跑自己的门；写实施记录（做了什么、为什么、与方案的出入、实测差异逐条定性） | commit + 记录 |
| 评审者 A（实现审） | 只读 diff；逐分支对旧行为；**自己实测**而不是只信实现者的门 | REVIEW 文件，末节签/不签 |
| 评审者 B（设计审） | 对照方案文/目标；裁决实现中偏离原文的取舍；看剩余工作是否仍在计划内 | 同上 |

约束：评审者 ≠ 实现者；设计审优先由签过方案的那个 agent 做（认得自己的必改）；当前会话的 agent 可以是任一角色。方案本身也先经设计审 R1…Rn 至签再动手（"开工前必改"）。

## 隔离

每个角色一份独立工作副本（git 用 worktree：`git worktree add --detach <dir> <commit>`），评审者的副本**只读**，每轮 `git checkout --detach <被审 commit>`；简报与评审文件放在副本根目录、不入库。共享的虚拟环境 / node_modules 用软链，简报里明写"禁止会改共享环境的命令"（如 `uv sync` / `npm install`）和"单次命令时限"。主线副本只做 ff 合并。

## 各 agent 的无头起法

通用：简报写成文件 `<ROLE>_BRIEF_<step>.md`，prompt 只放指针；`(nohup timeout <s> <cmd> > /tmp/<step>.log 2>&1 &)`；在跑与否用 `ps -eo pid,etime,cmd | grep "<cmd 片段>" | grep -v "bash -c"`（pgrep 会匹配到 wrapper）；判卡死看日志 mtime；**评审文件存在**才算写完，日志末尾常能提前看到结论；同一 agent 别并发两份评审，跑着一份就把下一份简报备好排队。

- **kimi**（慢：首 token ~18s，一份评审 30–90 分钟正常）
  `cd <副本> && kimi -p @BRIEF.md`
  `-p` 不能配 `--auto/-y`，权限走 `~/.kimi-code/config.toml` 的 `permission.rules`；单个 argv 上限 128KB，简报必须走文件；续会话 `-r <session_id>`（结束时 stdout 打印）；会派子代理，进度看 `~/.kimi-code/sessions/wd_*/session_*/agents/*/wire.jsonl` 的 mtime。
- **grok**（快：5–15 分钟）
  `grok --prompt-file BRIEF.md --always-approve -m <model> --reasoning-effort high --output-format plain --cwd <副本>`
  续会话 `-c`（同 cwd 最近会话）；写完文件后进程可能再挂几分钟，以文件为准。
- **claude**（当评审者时另起无头会话，别在当前会话里自审）
  `cd <副本> && claude -p "$(cat BRIEF.md)" --output-format text --dangerously-skip-permissions`
  续会话 `--resume <session-id>` 或 `-c`；限制工具用 `--allowedTools`。
- **codex**
  `codex exec -C <副本> --dangerously-bypass-approvals-and-sandbox -o REVIEW_CODEX_<step>.md "$(cat BRIEF.md)"`
  嵌套沙箱环境里内层 bwrap 起不来，必须 bypass；断流/失败只重试一次，再失败按超额处理、退出该评审不阻塞主线；`-m` 指定模型前先确认代理认识。
- **回看**：`cc-connect timer add --delay <N>m --desc "<step> 双审回看" --prompt "回看 <两份 REVIEW 路径>；写完则摘要签/不签与必改、按必改修、归档、处置写进记录；三方签→合主线并汇报；没写完报进度再设 15 分钟 timer。不要静默。"`

## 简报契约（两个评审者共用，只换清单侧重）

1. 角色与身份（实现审 / 设计审——设计审要提"你签过的方案"）、副本路径、被审 commit、**基线 commit**、只读约束与时限。
2. 范围：`git diff <base>..<tip>` + commit 列表；方案文哪一节；实施记录在哪；上一轮评审归档在哪。
3. 「请按顺序写」编号清单，每条**指到 file:line 或函数名**，并要求**自己实测**（"请自己跑 X 与基线对比，别只信门"）。
4. 实现者**最不确定的地方点名**让它核。
5. 可量化的验收一行（计数器 / 门的期望值），让它核账实。
6. 末条固定：新增内容里的复述、静默兜底、名字嗅探 / 与方案相悖之处。
7. 输出契约：`写到 <路径>/REVIEW_<ROLE>_<step>.md，最后一节只写「**签**」（可附 ≤3 处必改，file:line）或「**不签**，剩余阻塞：…」。`

## 读结论与归档

```bash
grep -n "^## " REVIEW.md; awk '/^## [^#]*(不签|签)/{f=1} f' REVIEW.md
```
必改逐条落；观察项能便宜收的当轮收并在处置里写明，不便宜的记入下一步。评审文件归档进仓库的 reviews 目录（`<who>_<step>_review.md`），处置段追加到实施记录：`**<who> <step>（归档路径，对象 <commit>）：签/不签**，必改逐条 + 观察项处置 + 未动理由`。三方签 → `git -C <主线副本> merge --ff-only <branch>`；多步叠在分支上时，基线是上一步的 tip，合并后 rebase 后续。

## 实现者的门（自己这一票怎么投）

- 只跑与改动直接相关的测试；全量只在合并前后台跑一次；失败清单在基线上复跑分"旧债 / 本步自造"，两类都修——重构不豁免旧债。
- **before/after 实测**：改前先把改动会触及的所有输出面 dump 成基线（每个面一个几十行脚本，走公共入口），改后再 dump，按（只在旧 / 只在新 / 值不同，键）分类，**每一类逐条定性**（预期改善 / 修正 / 跟随声明 / 回归）写进记录；除非真零，别写"零差异"。
- 有"只降不升"的计数器就重钉并看 diff：涨了先解释再接受。
- golden 红先看聚合层是否也变：聚合不变只有行变，多半是行数或命名；抖动怀疑跑两次。

## 通用坑

- heredoc 里有反引号或 `$` 必须 `<<'EOF'`，否则 shell 会执行它。
- 在装饰器和 `class` / `def` 之间插代码，装饰器会落到插入的东西上；插到装饰器前面。
- 用 grep 过滤测试输出时同时抓 `^FAILED` 和 `SUBFAILED`（子测试失败不带 FAILED 前缀）。
- 评审进程退出不等于文件落盘，反之亦然；以文件为准。
- 给带"未知键即拒绝"门的 schema 加字段，要同时登记门与显式传参，否则加载静默失败、下游全空。
- 缓存过的索引（按 id 回查）对后来嫁接的对象失效；问事实要带着对象问。
- 一次性接受计数器新基线前先看 diff，别把涨的钉住。
