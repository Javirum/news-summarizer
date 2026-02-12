"""SQLite database storage for processed articles."""
import sqlite3
from datetime import datetime, timezone
from config import Config


class ArticleDatabase:
    """Persistent storage for processed news articles."""

    def __init__(self, db_path=None):
        self.db_path = db_path or Config.DB_PATH
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        """Create the articles table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                source TEXT,
                url TEXT UNIQUE,
                summary TEXT,
                sentiment TEXT,
                published_at TEXT,
                processed_at TEXT
            )
        """)
        self.conn.commit()

    def save_article(self, result):
        """Save or update a single article by URL."""
        self.conn.execute("""
            INSERT OR REPLACE INTO articles
                (title, source, url, summary, sentiment, published_at, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result["title"],
            result["source"],
            result["url"],
            result["summary"],
            result["sentiment"],
            result.get("published_at"),
            datetime.now(timezone.utc).isoformat(),
        ))
        self.conn.commit()

    def save_articles(self, results):
        """Save multiple articles in a single transaction."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.executemany("""
            INSERT OR REPLACE INTO articles
                (title, source, url, summary, sentiment, published_at, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                r["title"],
                r["source"],
                r["url"],
                r["summary"],
                r["sentiment"],
                r.get("published_at"),
                now,
            )
            for r in results
        ])
        self.conn.commit()

    def get_article(self, url):
        """Get an article by URL. Returns dict or None."""
        row = self.conn.execute(
            "SELECT * FROM articles WHERE url = ?", (url,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_articles(self):
        """Get all articles ordered by processed_at DESC."""
        rows = self.conn.execute(
            "SELECT * FROM articles ORDER BY processed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def search_articles(self, keyword):
        """Search articles by keyword in title and summary."""
        pattern = f"%{keyword}%"
        rows = self.conn.execute(
            "SELECT * FROM articles WHERE title LIKE ? OR summary LIKE ? ORDER BY processed_at DESC",
            (pattern, pattern),
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self):
        """Return the total number of stored articles."""
        return self.conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    def close(self):
        """Close the database connection."""
        self.conn.close()
