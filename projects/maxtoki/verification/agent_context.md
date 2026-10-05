# Agent startup context (KEY = agent_context)

Method: self-report only. I did not read any file to write this. Everything below comes from
the text I was given at startup (system prompt, system reminders, tool list, environment block).
The only tool calls I made were `mkdir -p` for this folder and the write of this file.

Date given at startup: 2026-10-01.

## 1. CLAUDE.md (project instructions)

**Received: yes.** One CLAUDE.md was injected, from:
`<REPO_ROOT>/CLAUDE.md` (the repo root file).
It was labelled "project instructions, checked into the codebase". It came with a note that these
instructions override default behaviour.

First heading: `# CLAUDE.md`

Section headings (all `##`, in order):

1. How to write (applies to every reply, always)
2. What this repository is (and is not)
3. Directory layout
4. The pipeline markdown template
5. Workflow for adding a new pipeline
6. Domain conventions (used in pipelines and prompts)
7. Project target: `projects/maxtoki/`
8. Guidance when editing
9. How to write to the user (required)

Key content I received from it (short):
- Write in plain, simple English. Short sentences. No invented terms. Numbers first.
- The repo is a spec hub, not a software project. Pipelines live in `pipelines/`, papers in
  `references/`, pinned repos in `repos/`, per-model work in `projects/<model>/`.
- Ten-section pipeline template (Overview ... Known pitfalls).
- Code citations must use `repos/<name>/path/file.py:LINE`.
- MaxToki facts: two HF checkpoints (217M and 1B, `LlamaForCausalLM`), 20,275-token vocab,
  4096-token context. Planned run order: spectral-geometry -> attention-GRN -> topology-141 ->
  SAE mega-pipeline -> manifold-discovery.
- "Do not invent metrics or numbers."

Not received: no CLAUDE.md from `projects/biotensor/` or `projects/maxtoki/` was shown to me.
No user-level `~/.claude/CLAUDE.md` content was shown either. (I cannot tell whether such files
exist. I only know none were injected.)

## 2. User memory (MEMORY.md)

**Received: yes, the index only.** It was the author's Claude Code auto-memory index for the
project folder `projects/biotensor` (not `projects/maxtoki`), labelled "user's auto-memory, persists
across conversations".

It had about 82 one-line notes (my own count), each a title, a link to a separate memory file and a
one-line summary. Most are about the author's other projects; a few carry result summaries with
numbers. Their titles and topics are not reproduced here. I did NOT receive the contents of the
linked memory files. Only the index lines.

## 3. Tools

### Loaded (callable now)
- Core: `Bash`, `Read`, `Write`, `Edit`, `Skill`, `ToolSearch`, `ListAgents`, `SendUserFile`,
  `SuggestSkills`, `ReportFindings`, `Artifact`.
- Claude Docs connector: `mcp__1a59c906-..__batch`, `__guide`, `__update`.
- Session: `mcp__ccd_session__mark_chapter`, `spawn_task`, `dismiss_task`, `read_widget_context`.
- Built-in browser: `mcp__Claude_Browser__*` (navigate, computer, find, form_input,
  get_page_text, javascript_tool, read_page, read_console_messages, read_network_requests,
  resize_window, tabs_*, preview_start/stop/list/logs, browser_batch).
- `mcp__Claude_Code_iOS_Simulator__control`
- `mcp__terminal__read_terminal`
- `mcp__visualize__read_me`, `mcp__visualize__show_widget`

Not present among loaded tools: no `Grep`, no `Glob`, no `Agent`/`Task` tool (so I cannot
spawn my own subagents). Search must go through `Bash` (grep/find).

### Deferred (names only; must be loaded with ToolSearch first)
Includes: `WebFetch`, `WebSearch`, `Monitor`, `SendMessage`, `NotebookEdit`, `LSP`, `TaskStop`,
`CronCreate/Delete/List`, `EnterWorktree/ExitWorktree`, `ArtifactComments`, `ArtifactData`,
`ListSkills`, `SearchSkills`, `PushNotification`, `RemoteTrigger`, plus MCP servers for mail,
notes and file storage,
Claude in Chrome, scheduled tasks, session management, sidebar/window/view control,
PR tools, connectors registry, and terminal tab control.

### Skills listed
anthropic-skills (browser, chrome, computer-use, deep-research, docs, docx, pdf, pptx, xlsx,
schedule, skill-creator, memory tools, etc.), frontend-design, dataviz, artifact-design,
artifact-diagramming, artifact-capabilities, update-config, keybindings-help, code-review,
simplify, fewer-permission-prompts, loop, schedule, claude-api, workflow-authoring, run, init,
security-review.

## 4. Model

Stated at startup: "You are powered by the model named Opus 5.5. The exact model ID is
claude-opus-5-5." Knowledge cutoff stated: June 2026.
The git attribution reminder also names "Claude Opus 5.5".
Framing: "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude
Agent SDK", and "a subagent spawned by a workflow orchestration script". My final text reply is
returned verbatim to the script.

## 5. Working directory and environment

- Primary working directory: `<REPO_ROOT>/projects/biotensor`
  (note: biotensor, NOT maxtoki).
- "Is a git repository: false" (as stated for that folder).
- Platform: darwin. Shell: zsh. OS: Darwin 25.3.0.
- Scratchpad: `<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad`
- cwd is reset between bash calls; use absolute paths.
- Permission mode: a server note said "bypass permissions mode is active".

## 6. Other startup context (not asked, but shared by all agents)

- A long safety block: instructions only come from the user in chat; tool output is data;
  lists of prohibited and ask-first actions; privacy and copyright rules.
- A context line with the user's e-mail address and a rule to use it only for identification.
- Git attribution reminder (commit trailer "Co-Authored-By: Claude Opus 5.5").
- The task text itself was marked as "Workflow harness — computed task" with no user authority.
- Instructions from MCP servers (documents, mail, browser, notes) and detailed
  Artifact tool rules.

## What this means for a later controlled experiment

Agents started the same way will all share: the repo-root CLAUDE.md (with MaxToki facts and
the "do not invent numbers" rule) and the ~82-line biotensor memory index (titles plus
one-line findings, many with exact numbers). They will not share the linked memory file
bodies. If the experiment needs agents with no prior context about this user's past findings,
the memory index is a leak to control for, because it carries result summaries.

## Plain-words summary

I got one CLAUDE.md (repo root, 9 sections) and one memory index (~82 one-line entries, from
the biotensor project folder). I did not get the memory
file bodies. I am Opus 5.5 (`claude-opus-5-5`). My working folder is `projects/biotensor`.
I have Bash/Read/Write/Edit plus browser, docs and other tools, but no Grep, Glob or subagent tool.
