"""MkDocs hooks for circulax-specific preprocessing."""

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any


def on_startup(command: str, dirty: bool, **kwargs: Any) -> None:
    """Called once when the MkDocs build starts."""


def on_shutdown(**kwargs: Any) -> None:
    """Called once when the MkDocs build ends."""


def on_serve(server: Any, config: Any, builder: Any, **kwargs: Any) -> None:
    """Called when the MkDocs development server starts."""


def on_config(config: Any, **kwargs: Any) -> Any:
    """Called after config file is loaded but before validation."""
    return config


def on_pre_build(config: Any, **kwargs: Any) -> None:
    """Called before the build starts. Generates assets that must exist before file discovery."""
    _generate_lcr_animation()


def on_files(files: Any, config: Any, **kwargs: Any) -> Any:
    """Called after files are gathered but before processing."""
    return files


def on_nav(nav: Any, config: Any, files: Any, **kwargs: Any) -> Any:
    """Called after navigation is built."""
    return nav


def on_env(env: Any, config: Any, files: Any, **kwargs: Any) -> Any:
    """Called after Jinja2 environment is created."""
    return env


def on_post_build(config: Any, **kwargs: Any) -> None:
    """Called after the build is complete."""


def on_pre_template(template: Any, template_name: str, config: Any, **kwargs: Any) -> Any:
    """Called before a template is rendered."""
    return template


def on_template_context(context: Any, template_name: str, config: Any, **kwargs: Any) -> Any:
    """Called after template context is created."""
    return context


def on_post_template(output: str, template_name: str, config: Any, **kwargs: Any) -> str:
    """Called after template is rendered."""
    return output


def on_pre_page(page: Any, config: Any, files: Any, **kwargs: Any) -> Any:
    """Called before a page is processed."""
    return page


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any, **kwargs: Any) -> str:
    """Process markdown content before it's converted to HTML."""
    blocks = markdown.split("```")

    for i, block in enumerate(blocks):
        if i % 2:
            # This is a code block
            if (special := _parse_special(block)) is not None:
                blocks[i] = special
            else:
                blocks[i] = f"```{block}```"
            continue

        # This is regular markdown content
        lines = block.split("\n")
        _insert_cross_refs(lines)
        blocks[i] = "\n".join(lines)

    content = "".join(blocks)
    return content


def on_page_content(  # noqa: C901
    html: str, page: Any, config: Any, files: Any, **kwargs: Any
) -> str:
    """Called after markdown is converted to HTML."""
    if "```{svgbob}" not in html:
        return html

    source_parts = []
    for part in html.split("```"):
        if not part.startswith("{svgbob}"):
            continue
        lines = part.split("\n")[1:]
        for i, line in enumerate(lines):
            lines[i] = "".join(re.split(r"[<>]", line)[::2])
        part = dedent("\n".join(lines))
        source_parts.append(lines)

    rendered_parts = []
    for i, part in enumerate(html.split('<div class="language-text highlight">')):
        if i > 0:
            part = f'<div class="language-text highlight">{part}'
        first, *rest = part.split("</span></code></pre></div>")
        rest = "</span></code></pre></div>".join(rest)
        first = f"{first}</span></code></pre></div>"
        rendered_parts.append(first)
        rendered_parts.append(rest)

    for i, part in enumerate(rendered_parts):
        if not part.startswith('<div class="language-text highlight">'):
            continue
        lines = part.split("\n")
        if (svgbob_source := _svgbob_source(lines, source_parts)) is None:
            continue
        if (svg := _svgbob_svg(svgbob_source)) is None:
            continue
        rendered_parts[i] = svg

    return "".join(rendered_parts)


def on_page_context(context: Any, page: Any, config: Any, nav: Any, **kwargs: Any) -> Any:
    """Called after page context is created."""
    return context


def on_post_page(output: str, page: Any, config: Any, **kwargs: Any) -> str:
    """Called after page is fully processed."""
    return output


def _generate_lcr_animation() -> None:
    """Generate docs/images/lcr_animation.gif from the LCR transient simulation."""
    out_path = Path(__file__).parent / "images" / "lcr_animation.gif"
    if out_path.exists():
        return  # already generated; skip during incremental rebuilds

    import jax
    import jax.numpy as jnp
    import diffrax
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    from circulax import compile_circuit
    from circulax.components.electronic import Capacitor, Inductor, Resistor, VoltageSource
    from circulax.solvers import setup_transient

    out_path.parent.mkdir(parents=True, exist_ok=True)

    jax.config.update("jax_enable_x64", True)

    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "V1":  {"component": "source_voltage", "settings": {"V": 1.0, "delay": 0.25e-9}},
            "R1":  {"component": "resistor",        "settings": {"R": 10.0}},
            "C1":  {"component": "capacitor",       "settings": {"C": 1e-11}},
            "L1":  {"component": "inductor",        "settings": {"L": 5e-9}},
        },
        "connections": {
            "GND,p1": ("V1,p2", "C1,p2"),
            "V1,p1": "R1,p1",
            "R1,p2": "L1,p1",
            "L1,p2": "C1,p1",
        },
    }
    models = {
        "resistor": Resistor, "capacitor": Capacitor,
        "inductor": Inductor, "source_voltage": VoltageSource,
        "ground": lambda: 0,
    }

    circuit = compile_circuit(net_dict, models)
    y_op    = circuit()
    sim     = setup_transient(groups=circuit.groups, linear_strategy=circuit.solver)
    t_max   = 3e-9
    sol     = sim(
        t0=0.0, t1=t_max, dt0=1e-3 * t_max, y0=y_op,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0, t_max, 500)),
        max_steps=100_000,
    )

    ts    = sol.ts
    v_src = circuit.get_port_field(sol.ys, "V1,p1")
    v_cap = circuit.get_port_field(sol.ys, "C1,p1")
    i_ind = sol.ys[:, 5]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    (ln1,) = ax1.plot([], [], "b-",  linewidth=2.5, label="Capacitor V")
    (ln2,) = ax1.plot([], [], "g--", linewidth=2,   label="Source V")
    (ln3,) = ax2.plot([], [], "r:",  linewidth=2,   label="Inductor I")
    ax1.set_xlim(0, float(t_max))
    ax1.set_ylim(float(v_cap.min()) - 0.1, float(v_cap.max()) + 0.2)
    ax2.set_ylim(float(i_ind.min()) - 0.005, float(i_ind.max()) + 0.005)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (V)")
    ax2.set_ylabel("Current (A)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title("LCR Impulse Response — underdamped ringing at ~1 GHz")
    ax1.grid(True)
    fig.tight_layout()

    def _frame(i: int):
        n = max(2, i * 5)
        for ln, y in [(ln1, v_cap), (ln2, v_src), (ln3, i_ind)]:
            ln.set_data(ts[:n], y[:n])
        return ln1, ln2, ln3

    ani = animation.FuncAnimation(fig, _frame, frames=100, interval=40, blit=True)
    ani.save(str(out_path), writer="pillow", fps=25)
    plt.close(fig)


def _parse_special(content: str) -> str | None:
    """Format contents of a special code block differently."""
    lines = content.strip().split("\n")
    first = lines[0].strip()
    rest = lines[1:]
    if not (first.startswith("{") and first.endswith("}")):
        return None
    code_block_type = first[1:-1].strip()
    if code_block_type == "svgbob":
        source = "\n".join(rest)
        content_hash = hashlib.md5(source.encode()).hexdigest()
        svg_content = _svgbob_svg(source)
        if not svg_content:
            return None
        docs_dir = Path(__file__).parent
        svg_path = docs_dir / "assets" / "svgbob" / f"svgbob_{content_hash}.svg"
        svg_path.parent.mkdir(exist_ok=True, parents=True)
        svg_path.write_text(svg_content or "")
        return f"\n\n![{svg_path.name}](/sax/{svg_path.relative_to(docs_dir)})\n\n"
    return _format_admonition(code_block_type, rest)


def _format_admonition(admonition_type: str, lines: list[str]) -> str:
    """Format lines as an admonition."""
    if admonition_type == "hint":
        admonition_type = "info"
    ret = f"!!! {admonition_type}\n\n"
    for line in lines:
        ret += f"    {line.strip()}\n"
    return ret


def _svgbob_svg(source: str) -> str | None:
    source = source.replace("&lt;", "<").replace("&gt;", ">")
    svgbob = shutil.which("svgbob_cli")
    if not svgbob:
        print("Warning: svgbob_cli is not installed or not found in PATH.")  # noqa: T201
        return None
    content_hash = hashlib.md5(source.encode()).hexdigest()
    txt_filename = f"svgbob_{content_hash}.txt"
    temp_path = Path(tempfile.gettempdir()).resolve() / "svgbob" / txt_filename
    temp_path.parent.mkdir(exist_ok=True)
    try:
        temp_path.write_text(source)
        return subprocess.check_output(  # noqa: S603
            [
                svgbob,
                "--background",
                "#00000000",
                "--stroke-color",
                "grey",
                "--fill-color",
                "grey",
                str(temp_path),
            ]
        ).decode()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Error generating SVG with svgbob: {e}")  # noqa: T201
        return None
    finally:
        temp_path.unlink()


def _svgbob_source(rendered_lines: list[str], source_parts: list[list[str]]) -> str | None:
    for source_lines in source_parts:
        if all(sl.strip() in rl for sl, rl in zip(source_lines, rendered_lines, strict=False)):
            return "\n".join(source_lines)
    return None


def _insert_cross_refs(lines: list[str]) -> None:
    """Insert cross-references in the markdown lines."""
