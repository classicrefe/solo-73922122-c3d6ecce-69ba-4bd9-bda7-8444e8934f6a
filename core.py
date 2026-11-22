"""开批、抽样、录入与判定的核心业务逻辑。"""

import json
import random
import secrets
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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- 解析与校验

def parse_items_text(text: str):
    """解析清单文本, 每行一条: 「条目号,铭牌编号」(也接受空白分隔、中文逗号)。"""
    items = []
    for lineno, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
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
        items.append((entry_no, nameplate_no))
    if not items:
        raise ValidationError("清单为空, 至少需要 1 条")
    return items


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

def create_batch(conn, batch_no, limit_pct, checkpoints, items, seed=None):
    """开批。同批号已存在时沿用原方案, 不重新抽样。

    返回 (batch_row, created, notice)。
    """
    batch_no = (batch_no or "").strip()
    if not batch_no:
        raise ValidationError("批号不能为空")
    if len(batch_no) > 64:
        raise ValidationError("批号过长(不超过 64 字符)")
    pairs = normalize_items(items)
    validate_plan(limit_pct, checkpoints, len(pairs))

    existing = get_batch_by_no(conn, batch_no)
    if existing is not None:
        notice = f"批号 {batch_no} 已存在, 沿用原抽样方案, 未重新抽样。"
        stored = conn.execute(
            "SELECT entry_no, nameplate_no FROM items WHERE batch_id = ?",
            (existing["id"],)).fetchall()
        if sorted((r["entry_no"], r["nameplate_no"]) for r in stored) != sorted(pairs):
            notice += " 注意: 本次上传清单与原清单不一致, 仍以原方案为准。"
        return existing, False, notice

    seed = (seed or "").strip() or secrets.token_hex(8)
    n_max = checkpoints[-1]
    rng = random.Random(seed)

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO batches (batch_no, limit_pct, checkpoints, seed, size, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (batch_no, limit_pct, json.dumps(checkpoints), seed, len(pairs), now_iso()))
    batch_id = cur.lastrowid
    item_ids = []
    for entry_no, nameplate_no in pairs:
        cur.execute(
            "INSERT INTO items (batch_id, entry_no, nameplate_no) VALUES (?, ?, ?)",
            (batch_id, entry_no, nameplate_no))
        item_ids.append(cur.lastrowid)
    # 随机无放回抽出有序名单
    for pos, item_id in enumerate(rng.sample(item_ids, n_max), 1):
        cur.execute("UPDATE items SET position = ? WHERE id = ?", (pos, item_id))
    conn.commit()
    return get_batch(conn, batch_id), True, (
        f"已开批: 全批 {len(pairs)} 条, 无放回抽样 {n_max} 条(种子 {seed})。")


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
