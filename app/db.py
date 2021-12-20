from flask import current_app, g
import dataset


def get_db():
    if 'db' not in g:
        g.db = dataset.connect('sqlite:///' + current_app.config['DATABASE'])
    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()


def init_app(app):
    app.teardown_appcontext(close_db)
