"""Typer CLI entrypoint for Apah."""

import datetime
from pathlib import Path
import shutil
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

import typer
from rich.panel import Panel
from rich.table import Table

from apah.cli import __version__
from apah.cli.client import (
    ApahAPIError,
    ApahClient,
    ModelNotLoadedError,
    ServerNotRunningError,
)
from apah.cli.formatting import console, human_duration, human_size
from apah.registry.checksum import (
    compute_composite_checksum,
    compute_manifest_checksums,
    verify_manifest,
)
from apah.registry.internal_client import InternalRegistryClient
from apah.registry.manifest import (
    ModelManifest,
    get_latest,
    get_model_dir,
    list_versions,
    parse_model_identifier,
    sanitize_model_name,
)
from apah.registry.quant_detect import detect_quant_format

app = typer.Typer(
    name="apah",
    help="Apah: Custom LLM inference runtime for NVIDIA GPUs.",
    add_completion=True,
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"Apah version {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    version: Optional[bool] = typer.Option(
        None,
        "-v",
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show Apah version and exit.",
    ),
) -> None:
    """Apah: Custom LLM inference runtime for NVIDIA GPUs.

    Example Usage:
        apah serve
        apah pull Qwen/Qwen2.5-0.5B-Instruct:v1.0.0
        apah run Qwen/Qwen2.5-0.5B-Instruct "Why is the sky blue?"
        apah ps
        apah stop Qwen/Qwen2.5-0.5B-Instruct
    """
    pass


audit_app = typer.Typer(
    name="audit",
    help="Tamper-evident audit log verification and inspection utilities.",
    no_args_is_help=True,
)
app.add_typer(audit_app, name="audit")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host address to bind."),
    port: int = typer.Option(11500, "--port", help="Port to listen on."),
    gpu_mem_fraction: float = typer.Option(0.9, "--gpu-mem-fraction", help="GPU memory limit fraction."),
    idle_timeout: int = typer.Option(300, "--idle-timeout", help="Auto-unload idle timeout in seconds."),
    no_idle_unload: bool = typer.Option(False, "--no-idle-unload", help="Disable automatic idle model unloading."),
    audit_log_content: bool = typer.Option(
        False, "--audit-log-content", help="Log full prompt and completion text in audit logs (default: metadata only)."
    ),
) -> None:
    """Start the Apah inference server in the foreground (blocking)."""
    banner = (
        f"[bold cyan]Apah Inference Server[/bold cyan]\n"
        f"Running on [bold green]http://{host}:{port}[/bold green]\n"
        f"[dim]Idle timeout: {idle_timeout}s | Content Logging: {audit_log_content}[/dim]"
    )
    console.print(Panel(banner, title="Apah Server", expand=False))
    from apah.engine.server import run_server
    run_server(
        host=host,
        port=port,
        gpu_mem_fraction=gpu_mem_fraction,
        idle_timeout=idle_timeout,
        no_idle_unload=no_idle_unload,
        audit_log_content=audit_log_content,
    )


@audit_app.command("verify")
def audit_verify(
    log_path: Optional[str] = typer.Option(
        None, "--log-path", help="Path to audit log file or directory (defaults to ~/.apah/audit_logs)."
    ),
) -> None:
    """Verify SHA256 hash chain integrity of audit log files."""
    from pathlib import Path
    from apah.security.audit_log import verify_log_integrity

    target = Path(log_path) if log_path else Path.home() / ".apah" / "audit_logs"
    result = verify_log_integrity(target)

    if result.ok:
        console.print(
            f"[bold green]AUDIT LOG INTEGRITY VERIFIED (PASS): {result.total_entries} records checked. Hash chain is unbroken.[/bold green]"
        )
    else:
        console.print(f"[bold red]AUDIT LOG INTEGRITY FAILED (TAMPERING DETECTED):[/bold red]")
        if result.tampered_line:
            console.print(f"[red]First tampered entry detected at line {result.tampered_line}.[/red]")
        console.print(f"[red]Details: {result.error_message}[/red]")
        raise typer.Exit(code=1)


@audit_app.command("tail")
def audit_tail(
    lines: int = typer.Option(10, "-n", "--lines", help="Number of audit log records to display."),
    log_path: Optional[str] = typer.Option(
        None, "--log-path", help="Path to audit log file or directory (defaults to ~/.apah/audit_logs)."
    ),
) -> None:
    """Pretty-print the last N audit log events in human-readable format."""
    import json
    from pathlib import Path

    target = Path(log_path) if log_path else Path.home() / ".apah" / "audit_logs"
    log_files = []
    if target.is_file():
        log_files = [target]
    elif target.is_dir():
        log_files = sorted(target.glob("apah_audit_*.jsonl"))

    if not log_files:
        console.print("[yellow]No audit log files found.[/yellow]")
        return

    all_records = []
    for file in log_files:
        with open(file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        all_records.append(json.loads(line.strip()))
                    except Exception:
                        pass

    tail_records = all_records[-lines:] if lines > 0 else all_records

    if not tail_records:
        console.print("[yellow]No audit events found.[/yellow]")
        return

    table = Table(title=f"Last {len(tail_records)} Audit Events")
    table.add_column("TIMESTAMP", style="dim white", no_wrap=True)
    table.add_column("EVENT TYPE", style="bold cyan")
    table.add_column("ACTOR", style="green")
    table.add_column("DETAILS", style="yellow")
    table.add_column("HASH (SHORT)", style="dim magenta")

    for r in tail_records:
        ts = str(r.get("timestamp", "N/A"))[:19].replace("T", " ")
        event_type = str(r.get("event_type", "N/A"))
        actor = str(r.get("actor", "local"))
        details = json.dumps(r.get("details", {}))
        short_hash = str(r.get("entry_hash", ""))[:12]
        table.add_row(ts, event_type, actor, details, short_hash)

    console.print(table)


@app.command()
def run(
    model: str = typer.Argument(..., help="Model name or identifier (e.g. model:version)."),
    prompt: Optional[str] = typer.Argument(None, help="Optional text prompt for one-shot generation."),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream output tokens as they arrive."),
    tp: int = typer.Option(1, "--tp", help="Tensor parallel world size (number of GPUs)."),
    url: str = typer.Option("http://localhost:11500", "--url", help="Apah server base URL."),
) -> None:
    """Run model inference (one-shot prompt or interactive chat REPL)."""
    import torch
    if tp > 1 and torch.cuda.is_available():
        visible_gpus = torch.cuda.device_count()
        if tp > visible_gpus:
            console.print(f"[bold red]Error:[/bold red] Requested tensor parallel --tp {tp}, but only {visible_gpus} GPU(s) are visible.")
            raise typer.Exit(code=1)

    client = ApahClient(base_url=url)

    # Check server liveness
    try:
        ps_list = client.ps()
    except ServerNotRunningError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    # Resolve local model directory and manifest for load-time integrity verification
    clean_name, version = parse_model_identifier(model)
    manifest = get_latest(model)
    if manifest:
        model_dir = get_model_dir(clean_name, version=manifest.version)
        if model_dir.exists():
            verification = verify_manifest(model_dir, manifest)
            if not verification.ok:
                console.print(f"[bold red]Error: Pre-load integrity verification failed for '{model}':[/bold red]")
                console.print(f"[red]{verification.error_message}[/red]")
                console.print("[bold red]Refusing to load corrupted model into GPU memory.[/bold red]")
                raise typer.Exit(code=1)

    # Auto-load model if not currently active on server
    is_loaded = any(m.get("name") in (model, clean_name) for m in ps_list)
    if not is_loaded:
        with console.status(f"[bold yellow]Loading {model} (tp={tp})...[/bold yellow]"):
            try:
                client.load(model_path=model, tp_world_size=tp)
            except ApahAPIError as e:
                console.print(f"[bold red]Error loading model '{model}':[/bold red] {e}")
                raise typer.Exit(code=1)
        console.print(f"[bold green]Successfully loaded '{model}' (tp={tp})[/bold green]")

    # One-shot mode
    if prompt is not None:
        messages = [{"role": "user", "content": prompt}]
        try:
            if stream:
                for token in client.chat_completion_stream(model=model, messages=messages):
                    console.print(token, end="")
                console.print()
            else:
                resp = client.chat_completion(model=model, messages=messages, stream=False)
                content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                console.print(content)
        except (ModelNotLoadedError, ApahAPIError) as e:
            console.print(f"[bold red]Generation error:[/bold red] {e}")
            raise typer.Exit(code=1)

    # Interactive REPL mode
    else:
        console.print(f"[bold cyan]Apah Interactive Chat Mode ({model})[/bold cyan]")
        console.print("[dim]Press Ctrl+D or Ctrl+C to exit.[/dim]\n")
        messages = []
        while True:
            try:
                user_text = input(">>> ").strip()
                if not user_text:
                    continue
                messages.append({"role": "user", "content": user_text})

                if stream:
                    response_parts = []
                    for token in client.chat_completion_stream(model=model, messages=messages):
                        console.print(token, end="")
                        response_parts.append(token)
                    console.print()
                    messages.append({"role": "assistant", "content": "".join(response_parts)})
                else:
                    resp = client.chat_completion(model=model, messages=messages, stream=False)
                    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                    console.print(content)
                    messages.append({"role": "assistant", "content": content})
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Exiting chat.[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[bold red]Error during chat completion:[/bold red] {e}")


@app.command()
def pull(
    model: str = typer.Argument(..., help="Model identifier (e.g. Qwen/Qwen2.5-0.5B-Instruct:v1.0.0)."),
    source: str = typer.Option("hf", "--source", help="Download source: hf, local, or registry."),
    local_path: Optional[str] = typer.Option(None, "--local-path", help="Local directory path or registry base path."),
    revision: str = typer.Option("main", "--revision", help="Model revision or branch name."),
) -> None:
    """Pull model weights, version, compute checksums, and verify manifest integrity."""
    from apah.security.network_guard import check_airgap_pull_source, AirgapViolationError
    try:
        check_airgap_pull_source(source)
    except AirgapViolationError as e:
        console.print(f"[bold red]Airgap Violation:[/bold red] {e}")
        raise typer.Exit(code=1)

    clean_name, version = parse_model_identifier(model)
    target_version = version or "v1.0.0"
    target_dir = get_model_dir(clean_name, version=target_version)
    target_dir.mkdir(parents=True, exist_ok=True)

    if source == "registry":
        reg_endpoint = local_path or "http://localhost:11500"
        with console.status(f"[bold yellow]Pulling '{clean_name}:{target_version}' from registry at {reg_endpoint}...[/bold yellow]"):
            try:
                client = InternalRegistryClient(reg_endpoint)
                manifest = client.fetch_model(clean_name, version=target_version, dest_dir=target_dir)
                console.print(f"[bold green]Successfully pulled '{clean_name}:{target_version}' ({human_size(manifest.size_bytes)})[/bold green]")
                return
            except Exception as e:
                console.print(f"[bold red]Registry pull failed:[/bold red] {e}")
                raise typer.Exit(code=1)

    elif source == "local" and local_path:
        src_path = Path(local_path)
        if not src_path.exists():
            console.print(f"[bold red]Error:[/bold red] Local path '{local_path}' does not exist.")
            raise typer.Exit(code=1)
        with console.status(f"[bold yellow]Copying local model from {src_path}...[/bold yellow]"):
            for item in src_path.iterdir():
                dest = target_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

    elif source == "hf":
        try:
            from huggingface_hub import snapshot_download
            with console.status(f"[bold yellow]Pulling '{clean_name}' ({revision}) from Hugging Face...[/bold yellow]"):
                snapshot_download(repo_id=clean_name, local_dir=str(target_dir), revision=revision)
        except ImportError:
            console.print("[yellow]huggingface_hub not installed. Created target directory.[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Error pulling model from Hugging Face:[/bold red] {e}")
            raise typer.Exit(code=1)

    # Compute checksums, quant format, and metadata
    checksum_per_file = compute_manifest_checksums(target_dir)
    composite_sha256 = compute_composite_checksum(checksum_per_file)
    quant_format = detect_quant_format(target_dir)

    total_size = sum((target_dir / fname).stat().st_size for fname in checksum_per_file)

    manifest = ModelManifest(
        name=clean_name,
        version=target_version,
        source=source,
        revision=revision,
        size_bytes=total_size,
        quant=quant_format.value,
        architecture="unknown",
        checksum_sha256=composite_sha256,
        checksum_per_file=checksum_per_file,
    )

    manifest_file = target_dir / "apah_manifest.json"
    with open(manifest_file, "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    # Immediate integrity check
    verification = verify_manifest(target_dir, manifest)
    if not verification.ok:
        console.print(f"[bold red]Pull failed checksum verification:[/bold red] {verification.error_message}")
        raise typer.Exit(code=1)

    console.print(f"[bold green]Successfully pulled '{clean_name}:{target_version}' ({human_size(total_size)})[/bold green]")


@app.command(name="list")
def list_models() -> None:
    """List locally downloaded model manifests across all versions."""
    models_dir = Path.home() / ".apah" / "models"
    all_manifests = []

    if models_dir.exists():
        for p in models_dir.iterdir():
            if p.is_dir():
                manifests = list_versions(p.name, models_root=models_dir)
                all_manifests.extend(manifests)

    if not all_manifests:
        console.print("No local models found in ~/.apah/models/.")
        return

    table = Table(title="Local Models")
    table.add_column("NAME", style="cyan", no_wrap=True)
    table.add_column("VERSION", style="green")
    table.add_column("SIZE", style="magenta")
    table.add_column("QUANTIZATION", style="yellow")
    table.add_column("PULLED AT", style="blue")

    for m in all_manifests:
        table.add_row(
            m.name,
            m.version,
            human_size(m.size_bytes),
            m.quant,
            m.pulled_at[:19].replace("T", " "),
        )

    console.print(table)


@app.command()
def ps(
    url: str = typer.Option("http://localhost:11500", "--url", help="Apah server base URL."),
) -> None:
    """List models currently loaded in server GPU memory."""
    client = ApahClient(base_url=url)
    try:
        models = client.ps()
    except ServerNotRunningError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except ApahAPIError as e:
        console.print(f"[bold red]Error from server:[/bold red] {e}")
        raise typer.Exit(code=1)

    if not models:
        console.print("No models currently loaded on server.")
        return

    table = Table(title="Server Process Status")
    table.add_column("NAME", style="cyan", no_wrap=True)
    table.add_column("GPUs (TP)", style="bright_yellow", justify="right")
    table.add_column("GPU DEVS", style="dim cyan", no_wrap=True)
    table.add_column("GPU MEM", style="magenta", no_wrap=True)
    table.add_column("ACTIVE REQUESTS", style="green", justify="right")
    table.add_column("WAITING QUEUE", style="yellow", justify="right")
    table.add_column("IDLE", style="bright_magenta", no_wrap=True)
    table.add_column("UPTIME", style="blue", no_wrap=True)

    for m in models:
        gpu_mem_str = f"{m.get('gpu_memory_mb', 0.0):.1f} MB"
        uptime_str = human_duration(m.get("uptime_seconds", 0.0))
        idle_str = human_duration(m.get("idle_seconds", 0.0))
        devs = m.get("gpu_devices", [0])
        devs_str = ", ".join(str(d) for d in devs)
        table.add_row(
            str(m.get("name", "N/A")),
            str(m.get("tp_world_size", 1)),
            devs_str,
            gpu_mem_str,
            str(m.get("active_batch_size", 0)),
            str(m.get("waiting_queue_length", 0)),
            idle_str,
            uptime_str,
        )

    console.print(table)


@app.command()
def gpu(
    url: str = typer.Option("http://localhost:11500", "--url", help="Apah server base URL."),
) -> None:
    """Display real-time NVIDIA GPU hardware metrics (memory, utilization %, temperature)."""
    client = ApahClient(base_url=url)
    try:
        gpu_list = client.gpu()
    except ServerNotRunningError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except ApahAPIError as e:
        console.print(f"[bold red]Error from server:[/bold red] {e}")
        raise typer.Exit(code=1)

    if not gpu_list:
        console.print("[yellow]No NVML GPU statistics available from server.[/yellow]")
        return

    table = Table(title="NVIDIA GPU Hardware Telemetry")
    table.add_column("INDEX", style="cyan", justify="right")
    table.add_column("NAME", style="white", no_wrap=True)
    table.add_column("MEMORY (USED / TOTAL)", style="magenta", no_wrap=True)
    table.add_column("UTIL %", style="yellow", justify="right")
    table.add_column("TEMP (°C)", style="red", justify="right")

    for g in gpu_list:
        used_mb = float(g.get("memory_used_mb", 0.0))
        total_mb = float(g.get("memory_total_mb", 0.0))
        pct = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0
        used_bytes = int(used_mb * 1024 * 1024)
        total_bytes = int(total_mb * 1024 * 1024)
        mem_str = f"{human_size(used_bytes)} / {human_size(total_bytes)}"

        if pct < 70.0:
            styled_mem = f"[bold green]{mem_str}[/bold green]"
        elif pct < 90.0:
            styled_mem = f"[bold yellow]{mem_str}[/bold yellow]"
        else:
            styled_mem = f"[bold red]{mem_str}[/bold red]"

        table.add_row(
            str(g.get("index", 0)),
            str(g.get("name", "NVIDIA GPU")),
            styled_mem,
            f"{float(g.get('utilization_pct', 0.0)):.1f}%",
            f"{float(g.get('temperature_c', 0.0)):.1f}°C",
        )

    console.print(table)


@app.command()
def stop(
    model: Optional[str] = typer.Argument(None, help="Name of model to stop/unload."),
    url: str = typer.Option("http://localhost:11500", "--url", help="Apah server base URL."),
) -> None:
    """Unload the currently loaded model from server GPU memory."""
    client = ApahClient(base_url=url)
    try:
        res = client.unload(model_name=model)
        msg = res.get("message", "Model unloaded successfully.")
        console.print(f"[bold green]{msg}[/bold green]")
    except ServerNotRunningError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except ApahAPIError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def rm(
    model: str = typer.Argument(..., help="Model name or identifier (e.g. model:version)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Remove a local model directory or specific version from ~/.apah/models/."""
    clean_name, version = parse_model_identifier(model)
    manifest = get_latest(model)

    if not manifest:
        console.print(f"[bold red]Error:[/bold red] Model '{model}' not found in ~/.apah/models/.")
        raise typer.Exit(code=1)

    target_dir = get_model_dir(clean_name, version=version or manifest.version)

    if not yes:
        confirmed = typer.confirm(f"Are you sure you want to remove model '{clean_name}:{manifest.version}'?")
        if not confirmed:
            console.print("Aborted.")
            return

    shutil.rmtree(target_dir)
    console.print(f"[bold green]Successfully removed model '{clean_name}:{manifest.version}'[/bold green]")


@app.command()
def show(
    model: str = typer.Argument(..., help="Model name or identifier (e.g. model:version)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Display full per-file SHA256 checksum list."),
    url: str = typer.Option("http://localhost:11500", "--url", help="Apah server base URL."),
) -> None:
    """Show detailed manifest, checksum verification summary, and live server status."""
    clean_name, version = parse_model_identifier(model)
    manifest = get_latest(model)

    if manifest is None:
        console.print(f"[bold red]Error:[/bold red] Manifest for model '{model}' not found.")
        raise typer.Exit(code=1)

    model_dir = get_model_dir(clean_name, version=manifest.version)
    verification = verify_manifest(model_dir, manifest) if model_dir.exists() else None

    table = Table(title=f"Model Details: {manifest.name}:{manifest.version}", show_header=False)
    table.add_column("Property", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Name", manifest.name)
    table.add_row("Version", manifest.version)
    table.add_row("Source", manifest.source)
    table.add_row("Revision", manifest.revision)
    table.add_row("Pulled At", manifest.pulled_at)
    table.add_row("Size", human_size(manifest.size_bytes))
    table.add_row("Quantization", manifest.quant)

    files_count = len(manifest.checksum_per_file)
    if verification and verification.ok:
        integrity_str = f"[bold green]{files_count}/{files_count} files verified (OK)[/bold green]"
    elif verification:
        integrity_str = f"[bold red]FAILED ({len(verification.mismatched_files)} mismatched/missing)[/bold red]"
    else:
        integrity_str = "[yellow]Directory missing[/yellow]"

    table.add_row("Integrity", integrity_str)

    # Live server status check
    client = ApahClient(base_url=url)
    try:
        ps_list = client.ps()
        is_loaded = any(m.get("name") in (model, manifest.name) for m in ps_list)
        status_str = "[bold green]Loaded on server[/bold green]" if is_loaded else "[yellow]Not loaded[/yellow]"
    except Exception:
        status_str = "[dim]Server unreachable[/dim]"

    table.add_row("Live Status", status_str)
    console.print(table)

    if verbose and manifest.checksum_per_file:
        console.print("\n[bold cyan]Per-File SHA256 Checksums:[/bold cyan]")
        checksum_table = Table(show_header=True)
        checksum_table.add_column("File Path", style="cyan")
        checksum_table.add_column("SHA256 Checksum", style="dim white")
        for fname, hval in sorted(manifest.checksum_per_file.items()):
            checksum_table.add_row(fname, hval)
        console.print(checksum_table)


if __name__ == "__main__":
    app()
