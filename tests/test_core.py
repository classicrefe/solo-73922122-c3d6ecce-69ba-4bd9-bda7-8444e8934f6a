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
