# Vercel handler
from simple_app import app

# Vercel entry point
def handler(request):
    return app(request.environ, lambda status, headers: None)
