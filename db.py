"""SQLite 连接与建表。"""

import os
import sqlite3

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_no TEXT NOT NULL UNIQUE,          -- 批号
    limit_pct INTEGER NOT NULL CHECK (limit_pct BETWEEN 0 AND 99),  -- 错录率上限(%)
    checkpoints TEXT NOT NULL,              -- JSON 数组: 1~3 个递增的累计抽查数
    seed TEXT NOT NULL,                     -- 抽样随机种子(可审计复现)
    size INTEGER NOT NULL,                  -- 全批条目数 N
    next_checkpoint INTEGER NOT NULL DEFAULT 1,  -- 待判定的检查点序号(1 起)
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'released', 'pending_full')),
    -- 结批封存: 判定检查点、累计抽查数、实测错录数、超几何概率、阈值
    conclusion_checkpoint INTEGER,
    conclusion_n INTEGER,
    conclusion_errors INTEGER,
    conclusion_prob REAL,
    conclusion_threshold REAL,
    created_at TEXT NOT NULL,
    concluded_at TEXT
);

CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL REFERENCES batches(id),
    entry_no TEXT NOT NULL,                 -- 唯一条目号
    nameplate_no TEXT NOT NULL,             -- 铭牌编号
    position INTEGER,                       -- 抽样序号 1..n_max; NULL 表示未抽中
    result TEXT CHECK (result IN ('ok', 'err') OR result IS NULL),  -- 首个核对结果
    result_at TEXT,
    UNIQUE (batch_id, entry_no),
    UNIQUE (batch_id, position)
);

CREATE TABLE IF NOT EXISTS checkpoint_judgments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL REFERENCES batches(id),
    checkpoint_idx INTEGER NOT NULL,        -- 第几个检查点(1 起)
    n INTEGER NOT NULL,                     -- 累计抽查数
    errors INTEGER NOT NULL,                -- 实测错录数
    prob REAL NOT NULL,                     -- 超几何概率 P(X <= errors)
    threshold REAL NOT NULL,                -- 通过阈值 5% / 检查点数
    passed INTEGER NOT NULL,                -- 1 通过 / 0 未通过
    judged_at TEXT NOT NULL,
    UNIQUE (batch_id, checkpoint_idx)
);
"""


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        path = current_app.config["DB_PATH"]
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        g.db = conn
    return g.db


def close_db(_exc=None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db() -> None:
    conn = get_db()
    conn.executescript(SCHEMA)
    conn.commit()


def init_app(app) -> None:
    app.teardown_appcontext(close_db)
