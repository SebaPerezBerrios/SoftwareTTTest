TwitterExtractor
================

Instalación

```
source .venv/bin/activate &&
pip install -r requirements
```

El sistema funciona en base a SQLite para almacenar Tweets extraidos desde query especificada en settings.py como TRACK_TERMS

Para la autenticación de API de Twitter se requiere la creación del archivo private.py dentro de TwitterExtractor con la siguiente estructura.

```
TWITTER_APP_KEY=""
TWITTER_APP_SECRET=""
BEARER_TOKEN=""
```

Ejecución

```
python TwitterExtractor/__main__.py
```
