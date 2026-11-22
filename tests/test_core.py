"""核心逻辑与全流程自测。运行: python3 -m pytest tests/ -v"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from stats import hypergeom_cdf, min_exceeding_errors, pass_threshold


# ---------------------------------------------------------------- 统计函数

def naive_cdf(x, population, defects, draws):
    """用组合数直接计算的参照实现。"""
    lo = max(0, draws - (population - defects))
    hi = min(x, draws, defects)
    num = sum(math.comb(defects, k) * math.comb(population - defects, draws - k)
              for k in range(lo, hi + 1))
    return num / math.comb(population, draws)


def test_min_exceeding_errors():
    assert min_exceeding_errors(100, 5) == 6      # 100*5% = 5, 刚超限为 6
    assert min_exceeding_errors(50, 20) == 11
    assert min_exceeding_errors(10, 0) == 1       # 上限 0%, 刚超限为 1
    assert min_exceeding_errors(99, 99) == 99     # 99*0.99 = 98.01 -> 99
    assert min_exceeding_errors(7, 15) == 2       # 7*0.15 = 1.05 -> 2
    with pytest.raises(ValueError):
        min_exceeding_errors(10, 100)
    with pytest.raises(ValueError):
        min_exceeding_errors(10, -1)


def test_hypergeom_cdf_matches_naive():
    cases = [(10, 3, 4, 0), (10, 3, 4, 2), (20, 5, 10, 1), (50, 11, 25, 0),
             (50, 11, 25, 3), (60, 7, 50, 2), (9, 9, 9, 5)]
    for population, defects, draws, x in cases:
        assert hypergeom_cdf(x, population, defects, draws) == pytest.approx(
            naive_cdf(x, population, defects, draws), rel=1e-9, abs=1e-12)


def test_hypergeom_cdf_bounds():
    assert hypergeom_cdf(0, 10, 0, 5) == 1.0          # 无错录时必为 1
    assert hypergeom_cdf(5, 10, 3, 5) == 1.0          # x 覆盖上限时为 1
    assert hypergeom_cdf(-1, 10, 3, 5) == 0.0
    # 单点手算: N=10, D=3, n=1, P(X<=0) = 7/10
    assert hypergeom_cdf(0, 10, 3, 1) == pytest.approx(0.7)


def test_pass_threshold():
    assert pass_threshold(1) == pytest.approx(0.05)
    assert pass_threshold(2) == pytest.approx(0.025)
    assert pass_threshold(3) == pytest.approx(0.05 / 3)
    with pytest.raises(ValueError):
        pass_threshold(4)


# ---------------------------------------------------------------- 应用流程

@pytest.fixture()
def client(tmp_path):
    app = create_app({"TESTING": True, "DB_PATH": str(tmp_path / "t.db"),
                      "SECRET_KEY": "test"})
    return app.test_client()


def make_items(n):
    return [{"entry_no": f"E{i:03d}", "nameplate_no": f"NP-{i:03d}"}
            for i in range(1, n + 1)]


def open_batch(client, batch_no="B1", limit=20, cps=(25, 40), n=50, seed="s1"):
    return client.post("/api/batches", json={
        "batch_no": batch_no, "limit_pct": limit, "checkpoints": list(cps),
        "items": make_items(n), "seed": seed})


def sample_of(client, batch_no="B1"):
    resp = client.get(f"/api/batches/{batch_no}")
    assert resp.status_code == 200
    return resp.get_json()


def enter(client, entry_no, result, batch_no="B1"):
    return client.post(f"/api/batches/{batch_no}/results",
                       json={"entry_no": entry_no, "result": result})


def test_create_batch_draws_ordered_sample(client):
    resp = open_batch(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["created"] is True
    batch = body["batch"]
    assert batch["size"] == 50
    assert batch["assumed_errors"] == 11          # 50*20% = 10 -> 11
    assert batch["threshold"] == pytest.approx(0.025)
    assert batch["seed"] == "s1"
    sample = sample_of(client)["sample"]
    assert len(sample) == 40                       # 抽样数 = 最大累计抽查数
    positions = [it["position"] for it in sample]
    assert positions == list(range(1, 41))         # 有序名单
    assert len({it["entry_no"] for it in sample}) == 40   # 无放回


def test_same_seed_reproducible(client):
    open_batch(client, batch_no="A", seed="fixed")
    open_batch(client, batch_no="B", seed="fixed")
    a = [it["entry_no"] for it in sample_of(client, "A")["sample"]]
    b = [it["entry_no"] for it in sample_of(client, "B")["sample"]]
    assert a == b


def test_reupload_reuses_original_plan(client):
    open_batch(client, seed="first")
    before = sample_of(client)["sample"]
    resp = client.post("/api/batches", json={      # 同批号, 不同清单/种子/方案
        "batch_no": "B1", "limit_pct": 5, "checkpoints": [10],
        "items": make_items(30), "seed": "second"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["created"] is False
    assert "沿用原抽样方案" in body["notice"]
    assert "不一致" in body["notice"]
    after = sample_of(client)["sample"]
    assert [it["entry_no"] for it in before] == [it["entry_no"] for it in after]
    assert sample_of(client)["batch"]["limit_pct"] == 20   # 原方案不变


def test_validation_errors(client):
    def bad(**kw):
        payload = {"batch_no": "X", "limit_pct": 10, "checkpoints": [5],
                   "items": make_items(10)}
        payload.update(kw)
        resp = client.post("/api/batches", json=payload)
        assert resp.status_code == 400, payload
        return resp.get_json()["error"]

    assert "0~99" in bad(limit_pct=100)
    assert "1~3" in bad(checkpoints=[3, 6, 9, 12])
    assert "递增" in bad(checkpoints=[10, 10])
    assert "递增" in bad(checkpoints=[10, 5])
    assert "超过批量" in bad(checkpoints=[11])
    assert "重复" in bad(items=[{"entry_no": "E1", "nameplate_no": "A"},
                                {"entry_no": "E1", "nameplate_no": "B"}])
    assert "为空" in bad(items=[])
    assert "批号" in bad(batch_no="  ")


def test_pass_at_first_checkpoint_seals_conclusion(client):
    open_batch(client)                              # N=50, D=11, cps=[25,40]
    sample = sample_of(client)["sample"]
    first_leg = [it["entry_no"] for it in sample if it["position"] <= 25]
    rest = [it["entry_no"] for it in sample if it["position"] > 25]

    # 先录第二段(乱序): 检查点 1 前缀未齐, 不得判定
    for entry_no in rest:
        resp = enter(client, entry_no, "ok")
        assert resp.status_code == 200
        assert resp.get_json()["judgments"] == []

    # 再录第一段, 前 24 条也不应触发判定
    for entry_no in first_leg[:-1]:
        resp = enter(client, entry_no, "ok")
        assert resp.get_json()["judgments"] == []

    # 补齐到检查点 1: 判定并通过
    resp = enter(client, first_leg[-1], "ok")
    body = resp.get_json()
    assert body["status"] == "released"
    assert len(body["judgments"]) == 1
    j = body["judgments"][0]
    assert j["checkpoint_idx"] == 1 and j["n"] == 25 and j["errors"] == 0
    expected = hypergeom_cdf(0, 50, 11, 25)
    assert j["prob"] == pytest.approx(expected)
    assert j["prob"] <= 0.025

    # 结论已封存
    detail = sample_of(client)
    batch = detail["batch"]
    assert batch["status"] == "released"
    assert batch["conclusion_checkpoint"] == 1
    assert batch["conclusion_n"] == 25
    assert batch["conclusion_errors"] == 0
    assert batch["conclusion_prob"] == pytest.approx(expected)
    assert batch["concluded_at"]

    # 结批后停止录入
    resp = enter(client, first_leg[0], "err")
    assert resp.status_code == 409
    assert "停止录入" in resp.get_json()["error"]


def test_all_checkpoints_fail_marks_pending_full(client):
    open_batch(client)
    sample = sample_of(client)["sample"]
    judgments = []
    for it in sample:                               # 全部录为错误
        resp = enter(client, it["entry_no"], "err")
        judgments.extend(resp.get_json()["judgments"])
    assert [j["checkpoint_idx"] for j in judgments] == [1, 2]
    assert all(not j["passed"] for j in judgments)
    assert judgments[0]["errors"] == 25 and judgments[1]["errors"] == 40
    assert judgments[0]["prob"] == pytest.approx(1.0)
    batch = sample_of(client)["batch"]
    assert batch["status"] == "pending_full"
    assert batch["conclusion_checkpoint"] == 2
    assert batch["conclusion_n"] == 40
    resp = enter(client, sample[0]["entry_no"], "ok")
    assert resp.status_code == 409


def test_first_result_wins_and_unsampled_rejected(client):
    open_batch(client)
    sample = sample_of(client)["sample"]
    target = sample[0]
    assert enter(client, target["entry_no"], "err").status_code == 200
    resp = enter(client, target["entry_no"], "ok")   # 重复录入被拒, 首个结果保留
    assert resp.status_code == 409
    assert "首个结果" in resp.get_json()["error"]
    kept = [it for it in sample_of(client)["sample"]
            if it["entry_no"] == target["entry_no"]][0]
    assert kept["result"] == "err"

    unsampled = set(make_items(50)[i]["entry_no"] for i in range(50)) \
        - {it["entry_no"] for it in sample}
    entry_no = sorted(unsampled)[0]
    resp = enter(client, entry_no, "ok")
    assert resp.status_code == 409
    assert "未被抽中" in resp.get_json()["error"]

    assert enter(client, "NO-SUCH", "ok").status_code == 404
    assert client.post("/api/batches/NOPE/results",
                       json={"entry_no": "E001", "result": "ok"}).status_code == 404


def test_single_checkpoint_batch(client):
    open_batch(client, batch_no="S1", limit=20, cps=(30,))
    sample = sample_of(client, "S1")["sample"]
    assert len(sample) == 30
    for it in sample:
        resp = enter(client, it["entry_no"], "ok", batch_no="S1")
    body = resp.get_json()
    assert body["status"] == "released"
    assert body["judgments"][0]["threshold"] == pytest.approx(0.05)


def test_three_checkpoints_progression(client):
    # N=60, 上限 10% -> D=7; 检查点 [20,40,60], 阈值 0.05/3
    open_batch(client, batch_no="T3", limit=10, cps=(20, 40, 60), n=60)
    sample = sample_of(client, "T3")["sample"]
    by_pos = {it["position"]: it["entry_no"] for it in sample}
    # 第一段混 1 个错录: P(X<=1 | N=60,D=7,n=20) ≈ 0.247 > 阈值, 应不通过
    for pos in range(1, 21):
        enter(client, by_pos[pos], "err" if pos == 1 else "ok", batch_no="T3")
    detail = sample_of(client, "T3")
    assert len(detail["judgments"]) == 1
    j1 = detail["judgments"][0]
    assert j1["errors"] == 1 and not j1["passed"]
    assert j1["prob"] == pytest.approx(hypergeom_cdf(1, 60, 7, 20))
    assert detail["batch"]["status"] == "open"
    # 第二段全对: 累计错录仍为 1, n=40 时 P(X<=1) ≈ 0.0042, 应通过
    for pos in range(21, 41):
        enter(client, by_pos[pos], "ok", batch_no="T3")
    detail = sample_of(client, "T3")
    assert detail["batch"]["status"] == "released"
    j2 = detail["judgments"][1]
    assert j2["checkpoint_idx"] == 2 and j2["errors"] == 1 and j2["passed"]
    assert j2["prob"] <= 0.05 / 3
    assert detail["batch"]["conclusion_n"] == 40


def test_html_pages_render(client):
    open_batch(client)
    assert client.get("/batches").status_code == 200
    assert client.get("/batches/new").status_code == 200
    resp = client.get("/batches/B1")
    assert resp.status_code == 200
    assert "有序抽样名单".encode() in resp.data
    assert client.get("/batches/NOPE").status_code == 404


def test_html_form_flow(client):
    text = "\n".join(f"F{i:02d},NP-F{i:02d}" for i in range(1, 31))
    resp = client.post("/batches", data={
        "batch_no": "H1", "limit_pct": "20", "checkpoints": "15,25",
        "items_text": text, "seed": "html"}, follow_redirects=False)
    assert resp.status_code == 302
    detail = client.get("/api/batches/H1").get_json()
    assert len(detail["sample"]) == 25
    first = detail["sample"][0]["entry_no"]
    resp = client.post("/batches/H1/entry",
                       data={"entry_no": first, "result": "ok"})
    assert resp.status_code == 302
    kept = client.get("/api/batches/H1").get_json()["sample"][0]
    assert kept["result"] == "ok"


# ---------------------------------------------------------------- 流式上传与原子性

def _items_text(n):
    return "\n".join(f"E{i:05d},NP-{i:05d}" for i in range(1, n + 1))


def _post_file(client, text, batch_no="U1", seed="stream", checkpoints="20,40",
               limit="10", encoding="utf-8"):
    import io
    return client.post("/batches", data={
        "batch_no": batch_no, "limit_pct": limit, "checkpoints": checkpoints,
        "seed": seed,
        "list_file": (io.BytesIO(text.encode(encoding)), "list.txt")},
        content_type="multipart/form-data")


def test_streamed_upload_matches_textarea(client):
    text = _items_text(300)
    _post_file(client, text, batch_no="F1", seed="same")
    client.post("/batches", data={
        "batch_no": "F2", "limit_pct": "10", "checkpoints": "20,40",
        "seed": "same", "items_text": text})
    a = [it["entry_no"] for it in sample_of(client, "F1")["sample"]]
    b = [it["entry_no"] for it in sample_of(client, "F2")["sample"]]
    assert a == b
    assert sample_of(client, "F1")["batch"]["size"] == 300


def test_streamed_lines_crlf_and_no_final_newline(client):
    text = "E1,NP-1\r\nE2,NP-2\r\nE3,NP-3"     # CRLF 且末尾无换行
    resp = _post_file(client, text, batch_no="CR", checkpoints="2")
    assert resp.status_code == 302
    assert sample_of(client, "CR")["batch"]["size"] == 3


def test_upload_bad_line_rejects_whole_batch(client):
    text = _items_text(50) + "\n坏行只有一列\n"
    resp = _post_file(client, text, batch_no="BAD1")
    assert resp.status_code == 400
    assert "格式" in resp.get_data(as_text=True)
    assert client.get("/api/batches/BAD1").status_code == 404


def test_upload_invalid_utf8_rejected(client):
    import io
    resp = client.post("/batches", data={
        "batch_no": "UTF", "limit_pct": "10", "checkpoints": "5",
        "list_file": (io.BytesIO(b"E1,NP-1\n\xff\xfe\n"), "bad.txt")},
        content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "UTF-8" in resp.get_data(as_text=True)
    assert client.get("/api/batches/UTF").status_code == 404


def test_duplicate_entries_no_partial_write(client):
    text = "E1,NP-1\nE2,NP-2\nE1,NP-9\n"
    resp = _post_file(client, text, batch_no="DUP")
    assert resp.status_code == 400
    body = client.get("/api/batches").get_json()["batches"]
    assert all(b["batch_no"] != "DUP" for b in body)
    # 失败后同批号仍可正常开批
    assert _post_file(client, _items_text(50), batch_no="DUP").status_code == 302


def test_reupload_reordered_is_same_set(client):
    text = _items_text(40)
    _post_file(client, text, batch_no="R1", seed="rr")
    shuffled = "\n".join(reversed(text.splitlines()))
    resp = client.post("/api/batches", json={
        "batch_no": "R1", "limit_pct": 10, "checkpoints": [20, 40],
        "items": [{"entry_no": l.split(",")[0], "nameplate_no": l.split(",")[1]}
                  for l in shuffled.splitlines()], "seed": "other"})
    assert resp.status_code == 200
    notice = resp.get_json()["notice"]
    assert "沿用原抽样方案" in notice
    assert "不一致" not in notice            # 仅调换行序不算不同


def test_reupload_changed_pair_detected(client):
    _post_file(client, _items_text(40), batch_no="R2", seed="rr")
    items = [{"entry_no": f"E{i:05d}", "nameplate_no": f"NP-{i:05d}"}
             for i in range(1, 41)]
    items[7]["nameplate_no"] = "NP-CHANGED"   # 同一行序, 只改铭牌编号配对
    resp = client.post("/api/batches", json={
        "batch_no": "R2", "limit_pct": 10, "checkpoints": [20, 40],
        "items": items, "seed": "x"})
    assert "不一致" in resp.get_json()["notice"]
    # 原方案与原数据保持不变
    assert sample_of(client, "R2")["batch"]["seed"] == "rr"
    assert len(sample_of(client, "R2")["sample"]) == 40


def test_storage_failure_leaves_no_batch(client, monkeypatch):
    from core import StorageError

    def boom(conn, batch_id):
        raise StorageError("模拟写库失败")

    # 让抽样的第一次 UPDATE 就失败, 验证批次与条目整体回滚
    real_sample = __import__("random").Random.sample

    def exploding_sample(self, population, k):
        raise StorageError("模拟抽样阶段写库失败")

    monkeypatch.setattr("random.Random.sample", exploding_sample)
    resp = _post_file(client, _items_text(60), batch_no="FAIL1")
    assert resp.status_code == 500
    assert client.get("/api/batches/FAIL1").status_code == 404
    listed = client.get("/api/batches").get_json()["batches"]
    assert listed == []
    monkeypatch.setattr("random.Random.sample", real_sample)
    assert _post_file(client, _items_text(60), batch_no="FAIL1").status_code == 302


def test_api_storage_failure_no_batch(client, monkeypatch):
    def exploding(*_a, **_k):
        raise __import__("sqlite3").OperationalError("disk I/O error")

    monkeypatch.setattr("core._next_item_id", exploding)
    resp = client.post("/api/batches", json={
        "batch_no": "FAIL2", "limit_pct": 10, "checkpoints": [5],
        "items": make_items(10), "seed": "x"})
    assert resp.status_code == 500
    assert "写库失败" in resp.get_json()["error"]
    assert client.get("/api/batches/FAIL2").status_code == 404


# ---------------------------------------------------------------- 大清单与内存

def _stream_text(n, seed=0):
    """不生成中间列表的大清单文本(逐行 yield)。"""
    for i in range(1, n + 1):
        yield f"E{i:07d},NP-{i:07d}\n"


class _TextStream:
    def __init__(self, lines):
        self._lines = lines
        self._buf = ""

    def read(self, size=-1):
        while len(self._buf) < (size if size > 0 else 64 * 1024):
            try:
                self._buf += next(self._lines)
            except StopIteration:
                break
        out, self._buf = self._buf[:size], self._buf[size:]
        return out


def test_large_upload_line_boundaries(client):
    """大清单按块解码/切行, 不依赖一次性读入。"""
    from core import iter_text_lines

    def chunked(text, step):
        for i in range(0, len(text), step):
            yield text[i:i + step]

    text = "".join(_stream_text(5000))
    lines = list(iter_text_lines(chunked(text, 7)))
    assert len(lines) == 5000
    assert lines[0] == "E0000001,NP-0000001"
    assert lines[-1] == "E0005000,NP-0005000"


@pytest.mark.slow
def test_large_list_memory_bounded(tmp_path):
    """数十万条: 解析/校验/抽样/重传的额外内存不随行数线性增长。"""
    import sqlite3
    import tracemalloc

    import db as dbmod
    from core import create_batch, iter_upload_items

    conn = sqlite3.connect(str(tmp_path / "big.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA temp_store = FILE")
    conn.executescript(dbmod.SCHEMA)
    conn.commit()

    n = 300_000

    class LineStream:
        """逐行产出, 模拟上传文件流; 自身只保留一行。"""

        def __init__(self, order=None):
            self.i = 0
            self.order = order

        def read(self, size):
            if self.i >= n:
                return b""
            i = self.order[self.i] if self.order else self.i
            self.i += 1
            return f"E{i:07d},NP-{i:07d}\n".encode()

    tracemalloc.start()
    try:
        batch, created, _ = create_batch(
            conn, "BIG", 10, [100], iter_upload_items(LineStream()), seed="big")
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert created and batch["size"] == n
    assert peak < 64 * 1024 * 1024, peak

    # 重传同一份名单, 但行序完全打乱: 配对集合相同, 不新建、不提示不一致
    import random as _random
    order = list(range(n))
    _random.Random(1).shuffle(order)
    tracemalloc.start()
    try:
        _b2, created2, notice2 = create_batch(
            conn, "BIG", 10, [100], iter_upload_items(LineStream(order)), seed="other")
        _cur, peak2 = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert created2 is False
    assert "不一致" not in notice2
    assert peak2 < 64 * 1024 * 1024, peak2

    # 只改一个铭牌编号的配对 -> 提示不一致
    class ChangedStream(LineStream):
        def read(self, size):
            row = super().read(size)
            if row.startswith(b"E0000010,"):
                return b"E0000010,NP-CHANGED\n"
            return row

    _b3, created3, notice3 = create_batch(
        conn, "BIG", 10, [100], iter_upload_items(ChangedStream()), seed="x")
    assert created3 is False and "不一致" in notice3


def test_sample_matches_reference_draw():
    """相同种子下抽样结果必须与改造前(random.sample 于条目 id 序列)完全一致。"""
    import random
    import sqlite3
    import db as dbmod
    from core import create_batch

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA temp_store = FILE")
    conn.executescript(dbmod.SCHEMA)
    conn.commit()

    upload_order = list(range(1, 121))
    random.Random(7).shuffle(upload_order)
    pairs = [(f"E{i:03d}", f"NP-{i:03d}") for i in upload_order]
    batch, created, _ = create_batch(conn, "REF", 10, [30, 60], pairs, seed="ref-seed")
    assert created

    # 参照: 旧实现按上传顺序逐条 INSERT 得到连续 id, 再 rng.sample(ids, 60)
    first_id = conn.execute(
        "SELECT MIN(id) FROM items WHERE batch_id = ?", (batch["id"],)).fetchone()[0]
    expected = random.Random("ref-seed").sample(range(first_id, first_id + 120), 60)
    got = [r["id"] for r in conn.execute(
        "SELECT id FROM items WHERE batch_id = ? AND position IS NOT NULL "
        "ORDER BY position", (batch["id"],))]
    assert got == expected
    by_id = {r["id"]: r["entry_no"] for r in conn.execute(
        "SELECT id, entry_no FROM items WHERE batch_id = ?", (batch["id"],))}
    # 条目 id = first_id + 上传序号, 因此可比对抽中 id 对应到哪一条
    assert [by_id[i] for i in got] == [
        f"E{upload_order[i - first_id]:03d}" for i in got]
    # 条目 id 连续, 且与上传顺序一致
    rows = conn.execute(
        "SELECT id, entry_no FROM items WHERE batch_id = ? ORDER BY id", (batch["id"],)).fetchall()
    assert [r["id"] for r in rows] == list(range(first_id, first_id + 120))
    assert [r["entry_no"] for r in rows] == [f"E{i:03d}" for i in upload_order]


def test_reupload_still_validates_whole_file(client):
    """同批号重传也要逐行校验整份文件: 坏行/重复条目号一样整批拒绝, 原批次不受影响。"""
    _post_file(client, _items_text(40), batch_no="RV", seed="rv")
    before = sample_of(client, "RV")

    bad = "\n".join(_items_text(40).splitlines() + ["坏行"])
    resp = _post_file(client, bad, batch_no="RV")
    assert resp.status_code == 400
    assert "格式" in resp.get_data(as_text=True)

    dup = _items_text(40) + "\nE00001,NP-X"
    resp = _post_file(client, dup, batch_no="RV")
    assert resp.status_code == 400
    assert "重复" in resp.get_data(as_text=True)

    after = sample_of(client, "RV")
    assert [it["entry_no"] for it in before["sample"]] == \
           [it["entry_no"] for it in after["sample"]]
    assert after["batch"]["size"] == 40
    assert after["batch"]["seed"] == "rv"


def test_reupload_same_file_same_plan(client):
    """同一份文件重传: 沿用原方案, 不提示不一致, 抽样完全不变。"""
    text = _items_text(200)
    _post_file(client, text, batch_no="RS", seed="rs", checkpoints="50,100")
    first = [(it["position"], it["entry_no"]) for it in sample_of(client, "RS")["sample"]]
    resp = _post_file(client, text, batch_no="RS", seed="different", checkpoints="10")
    assert resp.status_code == 302
    detail = sample_of(client, "RS")
    assert detail["batch"]["seed"] == "rs"
    assert detail["batch"]["checkpoints"] == [50, 100]
    assert [(it["position"], it["entry_no"]) for it in detail["sample"]] == first
