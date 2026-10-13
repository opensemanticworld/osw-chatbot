"""Skills: longer instruction sets the assistant pulls in only when relevant.

A skill is a markdown file in ``skills/`` with yaml-ish front matter::

    ---
    name: osl-page-editing
    description: How to write wiki page content...
    ---

Only the one-line index (name + description) goes into the system prompt; the
body is fetched with the ``load_skill`` tool when the model decides it applies.
Keeping the bodies out of the base prompt matters - together they are far
longer than the rest of the prompt and most turns need neither.
"""

from dataclasses import dataclass
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"


@dataclass
class Skill:
    name: str
    description: str
    body: str


def _parse(text: str, fallback_name: str) -> Skill:
    name, description, body = fallback_name, "", text
    if text.startswith("---"):
        _, _, rest = text.partition("\n")
        front, sep, body = rest.partition("\n---")
        if sep:
            body = body.lstrip("\n")
            for line in front.splitlines():
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if key == "name" and value:
                    name = value
                elif key == "description" and value:
                    description = value
        else:
            body = text
    return Skill(name=name, description=description, body=body.strip())


def load_skills(directory=None):
    """Read every skill in `directory`, keyed by name. Never raises."""
    directory = Path(directory) if directory else SKILLS_DIR
    skills = {}
    if not directory.is_dir():
        return skills
    for path in sorted(directory.glob("*.md")):
        try:
            skill = _parse(path.read_text(encoding="utf-8"), path.stem)
        except Exception as e:  # noqa: BLE001 - a broken skill must not break the app
            print(f"osw-chatbot: skipping unreadable skill {path.name} ({e})")
            continue
        skills[skill.name] = skill
    return skills


def skill_index(skills) -> str:
    """The one-line-per-skill listing that goes into the system prompt."""
    if not skills:
        return ""
    lines = [
        "You have skills - detailed instructions you can read before a task. "
        "Call load_skill with the name as soon as a task matches one of them, "
        "and follow what it says:"
    ]
    for skill in skills.values():
        lines.append(f"- {skill.name}: {skill.description}")
    return "\n".join(lines)
