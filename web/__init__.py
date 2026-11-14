"""Flask application factory."""
from __future__ import annotations

from flask import Flask

from .views import DataStore, bp


def create_app(store: DataStore | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["STORE"] = store or DataStore.load()
    app.register_blueprint(bp)
    return app
