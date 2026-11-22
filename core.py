"""开批、抽样、录入与判定的核心业务逻辑。"""

import codecs
import json
import random
import re
import secrets
import sqlite3
from datetime import datetime, timezone

from stats import MAX_CHECKPOINTS, hypergeom_cdf, min_exceeding_errors, pass_threshold

STATUS_LABELS = {"open": "进行中", "released": "放行", "pending_full": "待全检"}
RESULT_LABELS = {"ok": "正确", "err": "错误"}


class DomainError(Exception):
    status = 400


class ValidationError(DomainError):
    status = 400


class NotFoundError(DomainError):
    status = 404


class ConflictError(DomainError):
    status = 409


class StorageError(DomainError):
    status = 500


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- 解析与校验

# 与 str.splitlines() 相同的行边界。发行版行长固定时, 流式解析只保留当前行。
_LINE_BREAK_RE = re.compile("\\r\\n|[\\n\\r\\v\\f\\x1c\\x1d\\x1e\\x85\\u2028\\u2029]")
_READ_CHUNK = 64 * 1024
STAGE_BATCH_SIZE = 1000
STAGING_TABLE = "upload_items"


def _parse_item_line(raw, lineno):
    """解析一行; 空行返回 None, 格式错误抛 ValidationError。"""
    line = raw.strip()
    if not line:
        return None
    line = line.replace("，", ",")
    if "," in line:
        left, right = line.split(",", 1)
    else:
        parts = line.split()
        if len(parts) != 2:
            raise ValidationError(f"清单第 {lineno} 行格式应为「条目号,铭牌编号」: {raw!r}")
        left, right = parts
    entry_no, nameplate_no = left.strip(), right.strip()
    if not entry_no or not nameplate_no:
        raise ValidationError(f"清单第 {lineno} 行条目号或铭牌编号为空")
    return entry_no, nameplate_no


def parse_items_text(text: str):
    """解析清单文本, 每行一条: 「条目号,铭牌编号」(也接受空白分隔、中文逗号)。

    供小清单/测试使用; 该函数会一次性返回全部条目, 大文件请用 iter_upload_items。
    """
    items = list(iter_text_items(text))
    if not items:
        raise ValidationError("清单为空, 至少需要 1 条")
    return items


def _decoded_chunks(stream):
    """把二进制流按 UTF-8 增量解码; 也接受已是 str 的文本流。"""
    decoder = codecs.getincrementaldecoder("utf-8")()
    while True:
        chunk = stream.read(_READ_CHUNK)
        if not chunk:
            break
        if isinstance(chunk, str):
            yield chunk
            continue
        try:
            text = decoder.decode(chunk)
        except UnicodeDecodeError:
            raise ValidationError("清单文件不是有效的 UTF-8 文本")
        if text:
            yield text
    try:
        tail = decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        raise ValidationError("清单文件不是有效的 UTF-8 文本")
    if tail:
        yield tail


def iter_text_lines(chunks):
    """把文本块迭代器切成行, 语义等同 str.splitlines(), 内存只保留未完成的一行。

    末尾暂缓一个孤立的 '\\r', 等下一个块确认它是否与 '\\n' 组成 CRLF。
    """
    buf = ""
    for chunk in chunks:
        buf += chunk
        start = 0
        limit = len(buf)
        if buf.endswith("\r"):
            limit -= 1
        for match in _LINE_BREAK_RE.finditer(buf, 0, limit):
            yield buf[start:match.start()]
            start = match.end()
        buf = buf[start:]
    # 流结束, 剩余的孤立 '\r' 也是行边界
    start = 0
    for match in _LINE_BREAK_RE.finditer(buf):
        yield buf[start:match.start()]
        start = match.end()
    if start < len(buf):
        yield buf[start:]


def parse_checkpoints(value):
    """解析检查点: 逗号分隔文本或整数列表, 1~3 个严格递增的正整数。"""
    if isinstance(value, str):
        raw = [p.strip() for p in value.replace("，", ",").split(",") if p.strip()]
        try:
            cps = [int(p) for p in raw]
        except ValueError:
            raise ValidationError("累计抽查数必须是整数")
    else:
        cps = list(value or [])
        if any(not isinstance(c, int) or isinstance(c, bool) for c in cps):
            raise ValidationError("累计抽查数必须是整数")
    if not 1 <= len(cps) <= MAX_CHECKPOINTS:
        raise ValidationError("检查点数量必须为 1~3 个")
    if any(c < 1 for c in cps):
        raise ValidationError("累计抽查数必须为正整数")
    if any(cps[i] >= cps[i + 1] for i in range(len(cps) - 1)):
        raise ValidationError("累计抽查数必须严格递增")
    return cps


def validate_plan(limit_pct, checkpoints, size: int) -> None:
    if not isinstance(limit_pct, int) or isinstance(limit_pct, bool) or not 0 <= limit_pct <= 99:
        raise ValidationError("错录率上限必须是 0~99 的整数")
    if size < 1:
        raise ValidationError("清单为空, 至少需要 1 条")
    if checkpoints[-1] > size:
        raise ValidationError(
            f"最大累计抽查数 {checkpoints[-1]} 超过批量 {size}, 无放回抽样不可能")


def normalize_items(items):
    """把 [(entry_no, nameplate_no), ...] 或 dict 列表统一成元组列表并校验唯一性。"""
    pairs = []
    for it in items or []:
        if isinstance(it, dict):
            pairs.append((str(it.get("entry_no", "")).strip(),
                          str(it.get("nameplate_no", "")).strip()))
        else:
            pairs.append((str(it[0]).strip(), str(it[1]).strip()))
    if not pairs:
        raise ValidationError("清单为空, 至少需要 1 条")
    for entry_no, nameplate_no in pairs:
        if not entry_no or not nameplate_no:
            raise ValidationError("条目号或铭牌编号为空")
    seen = set()
    for entry_no, _ in pairs:
        if entry_no in seen:
            raise ValidationError(f"条目号重复: {entry_no}")
        seen.add(entry_no)
    return pairs


def _api_pairs(items):
    """逐条产出 API 传入的条目, 不额外保存整份清单。"""
    for it in items or []:
        if isinstance(it, dict):
            yield (str(it.get("entry_no", "")).strip(),
                   str(it.get("nameplate_no", "")).strip())
        else:
            yield (str(it[0]).strip(), str(it[1]).strip())


def _text_pairs(chunks):
    """逐行解析清单文本块; 产出的顺序即原清单顺序(空行跳过)。"""
    for lineno, raw in enumerate(iter_text_lines(chunks), 1):
        pair = _parse_item_line(raw, lineno)
        if pair is not None:
            yield pair


def iter_text_items(text):
    """从整段文本逐行产出条目(较小清单 / 表单文本框)。"""
    return _text_pairs([text or ""])


def iter_upload_items(stream):
    """从上传文件流逐行产出条目; 只保留当前行与固定大小的读缓冲。"""
    return _text_pairs(_decoded_chunks(stream))


# ---------------------------------------------------------------- 查询

def get_batch_by_no(conn, batch_no: str):
    return conn.execute("SELECT * FROM batches WHERE batch_no = ?", (batch_no,)).fetchone()


def get_batch(conn, batch_id: int):
    return conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()


def list_batches(conn):
    return conn.execute("SELECT * FROM batches ORDER BY id DESC").fetchall()


def sample_items(conn, batch_id: int):
    """有序抽样名单(按抽样序号)。"""
    return conn.execute(
        "SELECT * FROM items WHERE batch_id = ? AND position IS NOT NULL ORDER BY position",
        (batch_id,)).fetchall()


def judgments_of(conn, batch_id: int):
    return conn.execute(
        "SELECT * FROM checkpoint_judgments WHERE batch_id = ? ORDER BY checkpoint_idx",
        (batch_id,)).fetchall()


def checkpoint_progress(conn, batch_id: int, checkpoints):
    """每个检查点: 已核/需核 与累计实测错录数。"""
    progress = []
    for idx, n in enumerate(checkpoints, 1):
        checked = conn.execute(
            "SELECT COUNT(*) AS c FROM items WHERE batch_id = ? AND position <= ? "
            "AND result IS NOT NULL", (batch_id, n)).fetchone()["c"]
        errors = conn.execute(
            "SELECT COUNT(*) AS c FROM items WHERE batch_id = ? AND position <= ? "
            "AND result = 'err'", (batch_id, n)).fetchone()["c"]
        progress.append({"idx": idx, "n": n, "checked": checked, "errors": errors})
    return progress


# ---------------------------------------------------------------- 开批

# 上传清单先落到 SQLite 临时表(磁盘), 不把整份清单留在进程内存中。
# 临时表按连接隔离, 不会与并发请求互相干扰。

def stage_pairs(conn, pairs):
    """逐条解析并写入临时表, 返回条目数。格式/空值错误就地抛出。

    每攒 STAGE_BATCH_SIZE 条批量写入一次; 缓冲区大小固定, 与总行数无关。
    迭代中途失败时, 已写入的行也一并回滚, 不留半份清单。
    """
    conn.execute(f"DROP TABLE IF EXISTS temp.{STAGING_TABLE}")
    conn.execute(
        f"CREATE TEMP TABLE {STAGING_TABLE} ("
        "seq INTEGER PRIMARY KEY, entry_no TEXT NOT NULL, nameplate_no TEXT NOT NULL)")
    conn.commit()

    count = 0
    buffer = []
    cur = conn.cursor()
    insert_sql = (f"INSERT INTO {STAGING_TABLE} (entry_no, nameplate_no) VALUES (?, ?)")

    def flush():
        if buffer:
            cur.executemany(insert_sql, buffer)
            buffer.clear()

    try:
        for entry_no, nameplate_no in pairs:
            if not entry_no or not nameplate_no:
                raise ValidationError("条目号或铭牌编号为空")
            count += 1
            buffer.append((entry_no, nameplate_no))
            if len(buffer) >= STAGE_BATCH_SIZE:
                flush()
        flush()
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return count


def drop_staged(conn):
    try:
        conn.execute(f"DROP TABLE IF EXISTS temp.{STAGING_TABLE}")
        conn.commit()
    except sqlite3.Error:
        pass


def _first_duplicate(conn):
    """返回按上传顺序最先出现的重复条目号, 便于提示定位。"""
    return conn.execute(
        f"SELECT entry_no FROM {STAGING_TABLE} GROUP BY entry_no "
        "HAVING COUNT(*) > 1 ORDER BY MIN(seq) LIMIT 1").fetchone()


def _upload_differs(conn, batch_id, count):
    """比较「条目号, 铭牌编号」配对集合; 行序、方案参数不参与比较。"""
    stored = conn.execute(
        "SELECT COUNT(*) AS c FROM items WHERE batch_id = ?", (batch_id,)).fetchone()["c"]
    if stored != count:
        return True
    missing = conn.execute(
        f"SELECT 1 FROM (SELECT entry_no, nameplate_no FROM {STAGING_TABLE} "
        "EXCEPT SELECT entry_no, nameplate_no FROM items WHERE batch_id = ?) LIMIT 1",
        (batch_id,)).fetchone()
    if missing is not None:
        return True
    extra = conn.execute(
        f"SELECT 1 FROM (SELECT entry_no, nameplate_no FROM items WHERE batch_id = ? "
        f"EXCEPT SELECT entry_no, nameplate_no FROM {STAGING_TABLE}) LIMIT 1",
        (batch_id,)).fetchone()
    return extra is not None


def _next_item_id(conn):
    """取 items 下一个 AUTOINCREMENT 值(与无显式 id 时一致)。"""
    seq = 0
    try:
        row = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'items'").fetchone()
        if row is not None:
            seq = row[0]
    except sqlite3.OperationalError:
        pass
    # 兜底: 若序列缺失但已有数据(如旧库), 取现有最大 id。
    max_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM items").fetchone()[0]
    return max(seq, max_id) + 1


def _reuse_existing(conn, existing, batch_no, count):
    """同批号重传: 返回原方案, 仅当配对集合变化时附加提示。"""
    notice = f"批号 {batch_no} 已存在, 沿用原抽样方案, 未重新抽样。"
    try:
        differs = _upload_differs(conn, existing["id"], count)
    except sqlite3.Error as exc:
        raise StorageError(f"读取临时清单失败: {exc}")
    if differs:
        notice += " 注意: 本次上传清单与原清单不一致, 仍以原方案为准。"
    return existing, False, notice


def create_batch(conn, batch_no, limit_pct, checkpoints, items, seed=None):
    """开批。同批号已存在时沿用原方案, 不重新抽样。

    items 可以是任意可迭代的 (条目号, 铭牌编号) 序列, 也可为生成器。
    开批与重传都会按原格式校验整份清单; 清单先落临时表, 因此额外内存
    不随清单条目数增长。返回 (batch_row, created, notice)。
    """
    batch_no = (batch_no or "").strip()
    if not batch_no:
        raise ValidationError("批号不能为空")
    if len(batch_no) > 64:
        raise ValidationError("批号过长(不超过 64 字符)")

    try:
        try:
            count = stage_pairs(conn, items)
        except sqlite3.Error as exc:
            raise StorageError(f"清单写入临时存储失败: {exc}")
        if count < 1:
            raise ValidationError("清单为空, 至少需要 1 条")
        try:
            duplicate = _first_duplicate(conn)
        except sqlite3.Error as exc:
            raise StorageError(f"读取临时清单失败: {exc}")
        if duplicate is not None:
            raise ValidationError(f"条目号重复: {duplicate['entry_no']}")
        validate_plan(limit_pct, checkpoints, count)

        existing = get_batch_by_no(conn, batch_no)
        if existing is not None:
            return _reuse_existing(conn, existing, batch_no, count)

        seed = (seed or "").strip() or secrets.token_hex(8)
        n_max = checkpoints[-1]
        rng = random.Random(seed)
        try:
            # 写锁内再查一次, 防止并发重传各自新建批次。
            conn.execute("BEGIN IMMEDIATE")
            existing = get_batch_by_no(conn, batch_no)
            if existing is not None:
                conn.rollback()
                return _reuse_existing(conn, existing, batch_no, count)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO batches (batch_no, limit_pct, checkpoints, seed, size, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (batch_no, limit_pct, json.dumps(checkpoints), seed, count, now_iso()))
            batch_id = cur.lastrowid
            # 条目 id 连续分配: 第一条为 first_id, 顺序与上传顺序一致。
            # 锁内读取, 和 AUTOINCREMENT 实际会分配的首个 id 一致。
            first_id = _next_item_id(conn)
            cur.execute(
                f"INSERT INTO items (id, batch_id, entry_no, nameplate_no) "
                f"SELECT ? + seq - 1, ?, entry_no, nameplate_no "
                f"FROM {STAGING_TABLE} ORDER BY seq", (first_id, batch_id))
            if cur.rowcount != count:
                raise sqlite3.IntegrityError(
                    f"写库条数不符: 期望 {count}, 实际 {cur.rowcount}")
            # 与改造前一致: 在同一条目的 id 序列上做无放回有序抽样。
            chosen = rng.sample(range(first_id, first_id + count), n_max)
            for pos, item_id in enumerate(chosen, 1):
                cur.execute("UPDATE items SET position = ? WHERE id = ? AND batch_id = ?",
                            (pos, item_id, batch_id))
            if conn.execute(
                    "SELECT COUNT(*) AS c FROM items WHERE batch_id = ? AND position IS NOT NULL",
                    (batch_id,)).fetchone()["c"] != n_max:
                raise StorageError("抽样名单写入不完整, 已回滚")
            conn.commit()
        except StorageError:
            conn.rollback()
            raise
        except sqlite3.Error as exc:
            conn.rollback()
            raise StorageError(f"写库失败, 批次未创建: {exc}")
        except BaseException:
            conn.rollback()
            raise
        return get_batch(conn, batch_id), True, (
            f"已开批: 全批 {count} 条, 无放回抽样 {n_max} 条(种子 {seed})。")
    finally:
        drop_staged(conn)


# ---------------------------------------------------------------- 录入与判定

def record_result(conn, batch_id: int, entry_no: str, result: str):
    """录入一条核对结果。乱序可; 每条只记首个结果; 结批后拒录。

    返回 (item_row, 新产生的判定列表)。
    """
    if result not in RESULT_LABELS:
        raise ValidationError("核对结果必须是 ok(正确) 或 err(错误)")
    batch = get_batch(conn, batch_id)
    if batch is None:
        raise NotFoundError("批次不存在")
    if batch["status"] != "open":
        raise ConflictError(f"批次已结批({STATUS_LABELS[batch['status']]}), 停止录入")
    entry_no = (entry_no or "").strip()
    item = conn.execute(
        "SELECT * FROM items WHERE batch_id = ? AND entry_no = ?",
        (batch_id, entry_no)).fetchone()
    if item is None:
        raise NotFoundError(f"条目号 {entry_no} 不在本批清单中")
    if item["position"] is None:
        raise ConflictError(f"条目 {entry_no} 未被抽中, 无需核对录入")
    if item["result"] is not None:
        raise ConflictError(
            f"条目 {entry_no} 已记录首个结果({RESULT_LABELS[item['result']]}), 不再更改")
    conn.execute("UPDATE items SET result = ?, result_at = ? WHERE id = ?",
                 (result, now_iso(), item["id"]))
    judgments = judge_checkpoints(conn, batch)
    conn.commit()
    return conn.execute("SELECT * FROM items WHERE id = ?", (item["id"],)).fetchone(), judgments


def judge_checkpoints(conn, batch):
    """从当前检查点起, 凡「抽样第 1 条到检查点」全部核完即判定。

    通过则封存放行结论; 全部检查点核完仍未通过则标为待全检。
    """
    cps = json.loads(batch["checkpoints"])
    m = len(cps)
    threshold = pass_threshold(m)
    size = batch["size"]
    assumed = min_exceeding_errors(size, batch["limit_pct"])
    idx = batch["next_checkpoint"]
    made = []
    while idx <= m:
        n = cps[idx - 1]
        checked = conn.execute(
            "SELECT COUNT(*) AS c FROM items WHERE batch_id = ? AND position <= ? "
            "AND result IS NOT NULL", (batch["id"], n)).fetchone()["c"]
        if checked < n:
            break  # 第 1 条到该检查点未全部核完, 不判定
        errors = conn.execute(
            "SELECT COUNT(*) AS c FROM items WHERE batch_id = ? AND position <= ? "
            "AND result = 'err'", (batch["id"], n)).fetchone()["c"]
        prob = hypergeom_cdf(errors, size, assumed, n)
        passed = prob <= threshold
        conn.execute(
            "INSERT INTO checkpoint_judgments "
            "(batch_id, checkpoint_idx, n, errors, prob, threshold, passed, judged_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (batch["id"], idx, n, errors, prob, threshold, int(passed), now_iso()))
        made.append({"checkpoint_idx": idx, "n": n, "errors": errors,
                     "prob": prob, "threshold": threshold, "passed": passed})
        if passed:
            conn.execute(
                "UPDATE batches SET status = 'released', conclusion_checkpoint = ?, "
                "conclusion_n = ?, conclusion_errors = ?, conclusion_prob = ?, "
                "conclusion_threshold = ?, concluded_at = ?, next_checkpoint = ? "
                "WHERE id = ?",
                (idx, n, errors, prob, threshold, now_iso(), idx, batch["id"]))
            break
        idx += 1
        if idx > m:
            conn.execute(
                "UPDATE batches SET status = 'pending_full', conclusion_checkpoint = ?, "
                "conclusion_n = ?, conclusion_errors = ?, conclusion_prob = ?, "
                "conclusion_threshold = ?, concluded_at = ?, next_checkpoint = ? "
                "WHERE id = ?",
                (m, n, errors, prob, threshold, now_iso(), idx, batch["id"]))
    conn.execute("UPDATE batches SET next_checkpoint = ? WHERE id = ?", (idx, batch["id"]))
    return made
