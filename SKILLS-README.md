# Run Claude Skills Locally on Your Mac (No Cloud Upload Required)

Anthropic recently announced [Skills for Claude](https://www.anthropic.com/news/skills) - reusable folders with instructions, scripts, and resources that make Claude better at specialized tasks. CodeRunner lets you run these skills **entirely on your local machine** in a sandboxed environment.

**What this means:** Claude can now process your files (documents, spreadsheets, presentations, images) using these specialized skills while keeping all data on your Mac. No uploads, complete privacy.

> **About CodeRunner:** [CodeRunner](https://github.com/instavm/coderunner) executes AI-generated code in a truly isolated sandboxed environment on your Mac using Apple's native containers.

## Why Run Skills Locally?

- **Privacy:** Process sensitive documents, financial data, or proprietary files without cloud upload
- **Full Control:** Skills execute in an isolated container with VM-level isolation
- **Compatibility:** Works with Claude Desktop, Gemini CLI, Qwen CLI, or any MCP-compatible tool
- **Extensibility:** Import Anthropic's official skills or create your own custom skills

## Prerequisites

* Mac with Apple Silicon (M1/M2/M3/M4/M5)
* Install the latest `coderunner` by running the `./install.sh` script from the [main repository](https://github.com/instavm/coderunner)

```shell
git clone https://github.com/instavm/coderunner.git
cd coderunner
chmod +x install.sh
./install.sh
```

Installation takes ~2 minutes. The MCP server will be available at `http://coderunner.local:8222/mcp`

## Setup: Connect Your AI Tool

CodeRunner is exposed as an MCP server and works with any MCP-compatible tool. All execution happens locally on your Mac.

### Example: Gemini CLI Configuration

Edit `~/.gemini/settings.json`:

```json
{
  "theme": "Default",
  "selectedAuthType": "oauth-personal",
  "mcpServers": {
    "coderunner": {
      "httpUrl": "http://coderunner.local:8222/mcp"
    }
  }
}
```

For system instructions, replace `~/.gemini/GEMINI.md` with [GEMINI.md](https://github.com/instavm/coderunner/blob/main/examples/gemini/GEMINI.md)

### Other Supported Tools

- **Claude Desktop:** See [configuration guide](https://github.com/instavm/coderunner#option-1-claude-desktop-integration)
- **Qwen CLI:** Configure similar to Gemini CLI
- **Any MCP client:** Point to `http://coderunner.local:8222/mcp`

## Example Use Cases

Once configured, you can ask your AI to:

- "Create a professional PowerPoint presentation from this markdown outline"
- "Extract all tables from these 10 PDFs and combine into one Excel spreadsheet"
- "Generate ASCII art logo for my project"
- "Fill out this tax form PDF with data from my CSV file"
- "Batch process these 100 images: crop to 16:9 and rotate 90 degrees"


## Adding New Skills

You can extend CodeRunner with additional skills in two ways:

### Option 1: Import Anthropic's Official Skills

Download skills from [Anthropic's skills repository](https://github.com/anthropics/skills/) and copy to:

```
~/.coderunner/assets/skills/user/<new-skill-folder>
```

**Available Official Skills:**
- Microsoft Word (docx)
- Microsoft PowerPoint (pptx)
- Microsoft Excel (xlsx)
- PDF manipulation
- Image processing
- Slack GIF creator
- And more...

### Skills Directory Structure

Here's an example with 4 imported skills:
```shell
/Users/manish/.coderunner/assets/skills/
├── public
│   ├── image-crop-rotate
│   │   ├── scripts
│   │   └── SKILL.md
│   └── pdf-text-replace
│       ├── scripts
│       └── SKILL.md
└── user
    ├── docx
    │   ├── docx-js.md
    │   ├── LICENSE.txt
    │   ├── ooxml
    │   ├── ooxml.md
    │   ├── scripts
    │   └── SKILL.md
    ├── pptx
    │   ├── html2pptx.md
    │   ├── LICENSE.txt
    │   ├── ooxml
    │   ├── ooxml.md
    │   ├── scripts
    │   └── SKILL.md
    ├── slack-gif-creator
    │   ├── core
    │   ├── LICENSE.txt
    │   ├── requirements.txt
    │   ├── SKILL.md
    │   └── templates
    └── xlsx
        ├── LICENSE.txt
        ├── recalc.py
        └── SKILL.md
```


### Option 2: Create Your Own Skills

Create a folder matching the structure shown above. The only mandatory file is `SKILL.md`. See [Anthropic's skills documentation](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview) for details.

**Quick Method:**
Ask Claude to generate a skill for you:

> "Can you write a skill which creates ASCII art of words?"

Claude will create the skill and offer a ZIP download. Place the ZIP file directly in `~/.coderunner/assets/skills/user` (no need to expand).

**Manual Method:**
Create your own skill folder structure:

```
~/.coderunner/assets/skills/user/my-custom-skill/
├── SKILL.md              # Documentation and usage examples
├── scripts/              # Your Python/bash scripts
│   └── process.py
└── requirements.txt      # (optional) Python dependencies
```

## How Folder Mapping Works

CodeRunner provides a compatibility layer that lets you run Claude's skills locally without modification:

**Path Translation:**
- Claude's path: `/mnt/user-data` → CodeRunner's path: `/app/uploads`
- Skills designed for Claude work locally without any changes

**Accessing Your Local Files:**
- Place files in `~/.coderunner/assets/outputs` on your Mac
- They become available to skills inside the container via volume mounts
- The mapping is automatic - skills can access your files without cloud upload

**Skill Structure:**
- No changes needed to imported skills
- Original folder hierarchy and file organization remain identical
- Import Claude skills and use them directly

## Live Demo: ASCII Art Skill

Here's a real example using Gemini CLI:

```
> /mcp

Configured MCP servers:

🟢 coderunner - Ready (5 tools)
  Tools:
  - execute_python_code
  - get_skill_file
  - get_skill_info
  - list_skills
  - navigate_and_get_all_visible_text

> can you generate ascii art for "CODERUNNER"

✦ I will generate the ASCII art you desire. First, I must survey my available skills.
✓  list_skills (coderunner MCP Server)

✦ I have located a relevant skill: ascii-art. I will now retrieve its instructions.
✓  get_skill_info (coderunner MCP Server) {"skill_name":"ascii-art"}
✦ Your ASCII art is ready:

   1   ____ ___  ____  _____ ____  _   _ _   _ _   _ _____ ____
   2  / ___/ _ \|  _ \| ____|  _ \| | | | \ | | \ | | ____|  _ \
   3 | |  | | | | | | |  _| | |_) | | | |  \| |  \| |  _| | |_) |
   4 | |__| |_| | |_| | |___|  _ <| |_| | |\\  | |\\  | |___|  _ <
   5  \____\___/|____/|_____|_| \_\\___/|_| \_|_| \_|_____|_| \_\
   6




Using: 1 GEMINI.md file | 3 MCP servers (ctrl+t to view)
```

**What happened:**
1. AI discovered available skills via `list_skills`
2. Found the relevant `ascii-art` skill
3. Retrieved skill instructions with `get_skill_info`
4. Executed the skill locally in the sandbox
5. Returned results - all without uploading any data to the cloud



From [@apple/container](https://github.com/apple/container/blob/main/docs/technical-overview.md):
> Each container has the isolation properties of a full VM, using a minimal set of core utilities and dynamic libraries to reduce resource utilization and attack surface.

## Available MCP Tools

When connected, CodeRunner exposes these tools to your AI:

- `execute_python_code` - Execute code in the sandbox
- `get_skill_file` - Read skill files
- `get_skill_info` - Get skill documentation
- `list_skills` - List all available skills
- `navigate_and_get_all_visible_text` - Web scraping with Playwright

## Learn More

- **Main Repository:** [github.com/instavm/coderunner](https://github.com/instavm/coderunner)
- **Anthropic Skills:** [github.com/anthropics/skills](https://github.com/anthropics/skills)
- **Skills Documentation:** [docs.claude.com/skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview)
- **Blog: Building Offline Workspace:** [instavm.io/blog/building-my-offline-workspace-part-2-skills](https://instavm.io/blog/building-my-offline-workspace-part-2-skills)
- **Report Issues:** [github.com/instavm/coderunner/issues](https://github.com/instavm/coderunner/issues)

## Contributing

We welcome contributions! If you create useful skills or improve the coderunner implementation, please share them with the community.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/instavm/coderunner/blob/main/LICENSE) file for details.
