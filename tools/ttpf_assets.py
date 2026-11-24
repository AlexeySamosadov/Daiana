"""Complete the rpf9dump texture-pool manifest during the assets build."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import NamedTuple
import zlib

# `python -I` implies `-P`, so the script's directory is not on sys.path; the sibling modules are next to it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bcpack
from file_mode import published_mode
import ttpf_mips as ttpf


# The modules that decode a full chain, next to this file. Both pass stamps hash them, so a pool completed by an
# older decoder is done again, as one completed with another calibration is.
DECODER_SOURCES = ("bcpack.py", "bcpack_tables.py", "bcpack_tables_bptc.py", "calibrated_texture.py",
                   "tex_headers.py", "ttpf_mips.py")


def decoder_digest():
    h = hashlib.sha256()
    for name in DECODER_SOURCES:
        data = Path(__file__).with_name(name).read_bytes()   # a missing source is an error, not an empty hash
        h.update(f"{name}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


def strict_stamp():
    # Ends in "-strict" (candidate() keeps such rows without a profile); its first 19 characters name the DDS.
    return f"v1-{decoder_digest()[:16]}-strict"


def load_profile(work):
    path = Path(work, "bcpack-calibration.json")
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return strict_stamp(), None
    except OSError as exc:
        raise ValueError(f"{path}: cannot read the calibration profile: {exc.strerror or exc}") from None
    try:
        # json.loads() on the bytes also detects a UTF-16 profile or one saved with a BOM.
        profile = json.loads(raw)
    except ValueError as exc:   # JSONDecodeError, or an encoding json cannot detect
        # The profile is hand-written (make_game says where it goes), so a typo names the file, not a bare
        # 'Expecting property name enclosed in double quotes'.
        raise ValueError(f"{path}: not a JSON calibration profile: {exc}") from None
    if not isinstance(profile, dict) or profile.get("version") != 1:
        raise ValueError(f"Unsupported calibration version in {path}")
    try:
        from calibrated_texture import native
        native()
    except (OSError, ImportError) as exc:
        raise RuntimeError("Calibrated texture export requires NumPy and the release build of "
                           "tools/bcpack_calibration (see its README.md)") from exc
    return "v1-" + hashlib.sha256(decoder_digest().encode() + raw).hexdigest(), profile


# BCPack initial history tables (tools/bcpack_history_calib.py output). BC6H and BC7 chunks that reference the
# format's built-in history only decode with them. A folder the user puts in <work> is used in every build (strict
# unless --bcpack-provisional); the automatic one is calibrated from the user's own pools, and since every entry it
# derives is provisional, it is made and used only with --bcpack-provisional.
HISTORY_DIR = "bcpack_history"
AUTO_HISTORY_DIR = "bcpack_history_auto"
AUTO_HISTORY_RECORD = "calibration.key"
AUTO_HISTORY_CODECS = "bc6h,bc7"
CALIBRATION_SOURCES = ("bcpack_history_calib.py",)


class Settings(NamedTuple):
    """What complete_pool() decodes with. A plain (stamp, profile) pair is the strict pass without history."""
    stamp: str                      # the pass stamp written to manifest["BCPackPass"]
    profile: dict | None            # the calibrated reconstruction profile, if any
    history: str | None = None      # BCPack history file or folder for tools/bcpack.py
    provisional: bool = False       # decode unconfirmed history entries (labelled "-provisional")
    fallback: str | None = None     # decoder record for rows decoded by ttpf_mips; None = the strict stamp


def history_files(path):
    """The JSON files a history path stands for, sorted: the file itself, or the *.json directly in a folder."""
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() == ".json" and p.is_file())
    return [path]


def history_digest(path):
    h = hashlib.sha256()
    for p in history_files(path):
        data = p.read_bytes()
        h.update(f"{p.name}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


def check_history(path):
    """Read the BC6H and BC7 tables once, so a broken history file is one build error, not one per texture."""
    import bcpack_tables_bptc as bptc
    if not history_files(path):
        raise ValueError(f"{path}: no BCPack history JSON files in this folder")
    try:
        bptc.load_bc6h(os.fspath(path), with_confirmed_bits=True)
        bptc.load_bc7(os.fspath(path))
        for codec in (1, 3, 4, 5):
            bptc.load_bc_history(codec, os.fspath(path))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"{path}: not a usable BCPack history table: {exc}") from None


def calibration_key(pools):
    """Changes when the calibration tool, the decoder or the set of pools (name and size) changes."""
    h = hashlib.sha256(decoder_digest().encode())
    for name in CALIBRATION_SOURCES:
        data = Path(__file__).with_name(name).read_bytes()
        h.update(f"{name}\0{len(data)}\0".encode())
        h.update(data)
    for p in sorted(os.path.abspath(p) for p in pools):
        h.update(f"{p}\0{os.path.getsize(p)}\n".encode())
    return h.hexdigest()


def run_calibration(pools, out):
    """Calibrate BC6H/BC7 history tables from the pools into the folder `out` (tools/bcpack_history_calib.py)."""
    import bcpack_history_calib as calib
    try:
        calib.run(["--out", os.fspath(out), "--codec", AUTO_HISTORY_CODECS, *map(os.fspath, pools)])
    except calib.CliError as exc:
        raise ValueError(f"BCPack history calibration failed: {exc}") from None


def auto_history(work, pools, log=print):
    """<work>/bcpack_history_auto, calibrated from `pools` once and again only when calibration_key() changes."""
    folder = Path(work, AUTO_HISTORY_DIR)
    key = calibration_key(pools)
    record = folder / AUTO_HISTORY_RECORD
    try:
        if record.read_text(encoding="utf-8").strip() == key:
            return folder
    except OSError:
        pass
    log(f"assets: calibrating BCPack BC6H/BC7 history from {len(pools)} texture pool(s) into {folder} "
        "(one time; it reads the pools and can take a while)")
    # Calibrated into a staging folder and switched in whole, so a stopped run leaves no half-written table behind.
    staging = Path(work, AUTO_HISTORY_DIR + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        run_calibration(pools, staging)
        (staging / AUTO_HISTORY_RECORD).write_text(key + "\n", encoding="utf-8")
        shutil.rmtree(folder, ignore_errors=True)
        os.replace(staging, folder)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return folder


def load_settings(work, *, provisional=False, pools=(), log=print):
    """The settings for complete_pool(): the calibration profile, plus BCPack history and the decode mode.

    History comes from <work>/bcpack_history when the user put one there, otherwise, with `provisional`, from the
    automatic calibration of `pools`. Without either, this is load_profile() unchanged. The pass stamp and the
    decoder record of rows decoded through ttpf_mips cover the history files and the mode, so changing either
    completes the pools again."""
    stamp, profile = load_profile(work)
    user = Path(work, HISTORY_DIR)
    if user.exists():
        history = user
    elif provisional and pools:
        history = auto_history(work, pools, log)
    else:
        history = None
    if history is None and not provisional:
        return Settings(stamp, profile)
    if history is not None:
        check_history(history)
    mode = "provisional" if provisional else "strict"
    h = hashlib.sha256(decoder_digest().encode())
    h.update(b"\0" + (history_digest(history) if history is not None else "").encode() + b"\0" + mode.encode())
    fallback = f"v1-{h.hexdigest()[:16]}-{mode}"
    if profile is not None:
        stamp = "v1-" + hashlib.sha256(f"{stamp}\0{fallback}".encode()).hexdigest()
    else:
        stamp = fallback
    return Settings(stamp, profile, None if history is None else os.fspath(history), provisional, fallback)


def publish(path, data):
    f = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    tmp = Path(f.name)
    try:
        with f:
            f.write(data)
        if os.name != "nt":
            tmp.chmod(published_mode(path))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def file_identity(path):
    """(device, inode) of an existing file, or None: two names with one identity are one file on disk."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino) if st.st_ino else None


def remove_replaced(directory, replaced, rows):
    """Remove the files of `replaced` (names the rows had before they were switched) from `directory`; returns
    the names that could not be removed (in use), sorted.

    Only a .dds file by its bare name in this folder is removed, never manifest.json, and never a file a row still
    names: by the same name, by a name that differs in letter case only, or by any other spelling of the same file
    on disk (on Windows and macOS that can be the file the pass has just published). Each file is looked at once:
    one stat per named row and one per replaced name, not one per pair."""
    names = [row["File"] for row in rows if isinstance(row.get("File"), str)]
    named = {os.path.normcase(name) for name in names}
    on_disk = {file_identity(directory / name) for name in names} - {None}
    left = []
    for name in sorted(replaced):
        if Path(name).name != name or not name.lower().endswith(".dds") or os.path.normcase(name) in named:
            continue
        if file_identity(directory / name) in on_disk:
            continue
        try:
            (directory / name).unlink(missing_ok=True)
        except OSError:
            left.append(name)
    return left


# The manifest key that lists replaced files still to be removed. It is written with the manifest that switches the
# rows, so a removal that fails (a file in use) or a run stopped right after the switch is finished by a later run,
# even though that run finds the pass stamped and writes nothing else.
PENDING = "BCPackReplaced"


def finish_replaced(directory, manifest):
    """Remove the files the manifest lists under PENDING, and publish it again with what is left."""
    pending = manifest.get(PENDING)
    if not isinstance(pending, list) or not pending:
        return
    left = remove_replaced(directory, {name for name in pending if isinstance(name, str)}, manifest["Textures"])
    if left:
        manifest[PENDING] = left
    else:
        manifest.pop(PENDING)
    if left != pending:
        try:
            publish(directory / "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
        except OSError as exc:
            # The files are gone; the list on disk still names them, and a later run drops them (missing is fine).
            print(f"assets: {directory.name}: could not update manifest.json after removing replaced DDS files: "
                  f"{exc.strerror or exc}", file=sys.stderr)
    if left:
        print(f"assets: {directory.name}: {len(left)} replaced DDS file(s) could not be removed (in use?) and stay "
              "next to the full-chain files; the next build tries again", file=sys.stderr)


def candidate(row, profile, provisional=False):
    if row.get("DxgiFormat") not in ttpf.BC_FORMATS:
        return False
    if row.get("BCPackDecoder"):
        # A strict result is redone by any later pass; a provisional one only by another provisional pass (a strict
        # pass would refuse it again); a calibrated one only while there is a profile.
        decoder = row["BCPackDecoder"]
        return (profile is not None or decoder.endswith("-strict")
                or (provisional and decoder.endswith("-provisional")))
    return (row["File"] is None or row.get("BaseMipLevel", 0) > 0
            or row.get("Mips", 0) < row.get("SourceMips", 0))


def complete_pool(source, directory, manifest, settings):
    """Publish full chains next to ordinary DDS files, then switch the catalog to them."""
    settings = Settings(*settings)
    stamp, profile = settings.stamp, settings.profile
    directory = Path(directory)
    if manifest.get("BCPackPass") == stamp:
        finish_replaced(directory, manifest)
        return 0, 0
    rows = [row for row in manifest["Textures"] if candidate(row, profile, settings.provisional)]
    if not rows:
        return 0, 0
    written = blocked = 0
    superseded = set()   # the files the rows named before they were switched to a full chain
    with closing(ttpf.Pool(source)) as pool:
        decoder = strict = None
        if profile is not None:
            from calibrated_texture import LocalDecoder
            decoder = LocalDecoder(pool, profile)
        try:
            for row in rows:
                index = row["Index"]
                desc = pool.descriptor(index)
                base = desc["base"]
                if not pool.chunks or base < pool.chunks[0][0] or base >= pool.virtual_size:
                    continue
                span = pool.span(index, base)
                if not span:
                    continue
                try:
                    if not pool.codecs(base, span) & bcpack.SUPPORTED_REBUILD_CODECS:
                        continue
                    if decoder and desc["dimension"] == 1 and desc["format"] in decoder.formats:
                        method = stamp
                        blob = decoder.dds(index)
                    else:
                        if settings.fallback is not None:
                            method = settings.fallback
                        else:
                            if profile is not None and strict is None:
                                strict = strict_stamp()   # once per pass: it reads and hashes six sources
                            method = strict or stamp
                        analysis = ttpf.analyze(pool, index)
                        blob = ttpf.texture_dds(pool, analysis, allow_provisional=settings.provisional,
                                                history=settings.history)
                except (ValueError, KeyError, IndexError, ttpf.BcpackRequired,
                        bcpack.BcpackUnsupportedCodec, zlib.error) as exc:
                    row["BCPackLimitation"] = str(exc)
                    blocked += 1
                    continue
                # A distinct name keeps an existing lower-mip DDS usable until the manifest switches.
                name = f"{index:06d}_bcpack_{method[:19]}.dds"
                publish(directory / name, blob)
                if isinstance(row.get("File"), str) and row["File"] != name:
                    superseded.add(row["File"])
                row.update(File=name, Mips=desc["mips"], BaseMipLevel=0, Limitation=None, BCPackDecoder=method)
                row.pop("BCPackLimitation", None)
                written += 1
        finally:
            if decoder:
                decoder.read_chunk.cache_clear()
    manifest["BCPackPass"] = stamp
    # The files the switched rows named before would otherwise stay in the folder, named by no row, so every
    # completed texture would be stored twice. They are listed in the manifest that switches the rows and removed
    # only after it is published: a file another row still names is kept, and one that can't be removed (in use)
    # stays listed for the next build. The export is complete either way.
    pending = {name for name in manifest.get(PENDING) or [] if isinstance(name, str)} | superseded
    if pending:
        manifest[PENDING] = sorted(pending)
    publish(directory / "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    finish_replaced(directory, manifest)
    return written, blocked
