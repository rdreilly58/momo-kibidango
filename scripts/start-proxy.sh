#!/bin/bash
# Start the cascade proxy service
export ANTHROPIC_API_KEY=$(cat ~/.openclaw/credentials/anthropic)
export PYTHONPATH="/Users/rreilly/momo-kibidango/src:$PYTHONPATH"
cd /Users/rreilly/momo-kibidango
source venv/bin/activate
exec python -c "
from momo_kibidango.proxy import create_proxy_app
app = create_proxy_app()
app.run(host='127.0.0.1', port=7780, debug=False)
"
